"""
modules/pdf_intelligence.py  — v3 (fix: entities always empty on Streamlit Cloud)

Root causes fixed:
  1. Single LLM call tried to produce summary + entities + signals + type_specific
     + judge in one shot. With max_tokens=2500 the JSON was routinely truncated,
     causing json.loads() to fail silently → _empty_analysis() → empty entities.

  2. bare `except Exception: return None` hid every failure.

  3. On Streamlit Cloud the gpt-4o-mini response for a 25-field Legal doc with
     verbatim source_text snippets easily hits 2500 tokens.

Fixes:
  • Split into TWO cheaper calls:
      Call A — entities + signals  (max_tokens=3500)
      Call B — summary + judge     (max_tokens=1000)
  • JSON repair: if json.loads() fails, attempt to close truncated JSON before
    giving up (handles the single most common cloud failure mode).
  • Debug mode: set env var PDF_INTEL_DEBUG=1 to surface raw LLM responses in
    st.session_state["_pdf_intel_debug"] for inspection.
  • entities prompt asks for azure_di_key so UI can do exact bbox lookup.
"""

from __future__ import annotations

import json
import os
import re
import textwrap


# ─────────────────────────────────────────────────────────────────────────────
# AZURE OPENAI CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def _get_secrets() -> dict:
    """Read secrets from Streamlit Cloud or fallback to os.environ."""
    try:
        import streamlit as st
        return st.secrets
    except Exception:
        return os.environ

def _get_openai_client():
    try:
        from openai import AzureOpenAI
        s = _get_secrets()
        return AzureOpenAI(
            azure_endpoint=s.get("OPENAI_DEPLOYMENT_ENDPOINT", ""),
            api_key=s.get("OPENAI_API_KEY", ""),
            api_version=s.get("OPENAI_API_VERSION", "2024-12-01-preview"),
        )
    except Exception:
        return None

def _deployment() -> str:
    return _get_secrets().get("OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
   


# ─────────────────────────────────────────────────────────────────────────────
# JSON REPAIR  — handle truncated responses from token-limit hits
# ─────────────────────────────────────────────────────────────────────────────

def _repair_json(raw: str) -> str:
    """
    Attempt to close truncated JSON so json.loads() can succeed.
    Handles the most common truncation pattern: object cut off mid-string or
    mid-value while iterating over entities.
    """
    raw = raw.strip()

    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    # Try as-is first
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    # Count open braces / brackets to figure out what needs closing
    # Walk character by character tracking open delimiters
    stack: list[str] = []
    in_str    = False
    escape    = False
    for ch in raw:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in ("{", "["):
            stack.append("}" if ch == "{" else "]")
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()

    # Close any unterminated string
    if in_str:
        raw += '"'

    # Close any trailing incomplete key-value (ends with ": " or ": {")
    # Strip trailing comma before closing
    raw = re.sub(r",\s*$", "", raw.rstrip())

    # Close open containers in reverse
    closing = "".join(reversed(stack))
    repaired = raw + closing

    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        pass

    return raw  # give up — caller will handle the parse error


def _llm_call(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 3500,
    label: str = "llm_call",
) -> dict | None:
    """
    Call the LLM and return parsed JSON, or None on failure.
    Stores raw response in session state when PDF_INTEL_DEBUG=1.
    """
    client = _get_openai_client()
    if not client:
        _debug_store(label, "ERROR: no client (check OPENAI env vars)")
        return None

    try:
        response = client.chat.completions.create(
            model=_deployment(),
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        _debug_store(label, raw)

        repaired = _repair_json(raw)
        return json.loads(repaired)

    except json.JSONDecodeError as e:
        _debug_store(label + "_parse_error", str(e))
        return None
    except Exception as e:
        _debug_store(label + "_error", str(e))
        return None


def _debug_store(key: str, value: str) -> None:
    if os.environ.get("PDF_INTEL_DEBUG", "0") != "1":
        return
    try:
        import streamlit as st
        bucket = st.session_state.setdefault("_pdf_intel_debug", {})
        bucket[key] = value
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_full_text_from_parsed(parsed: dict) -> str:
    parts: list[str] = []
    for page in parsed.get("pages", []):
        raw = (page.get("raw_text") or "").strip()
        if raw:
            parts.append(f"[PAGE {page['page_num']}]\n{raw}")
    return "\n\n".join(parts)


def build_azure_di_field_index(sheet_cache: dict) -> dict[str, dict]:
    """
    Flatten all Azure DI fields from sheet_cache into a single lookup:
        { field_name: { value, confidence, bounding_polygon, source_page,
                        page_width, page_height } }
    Used both to feed the LLM prompt (name→value map) and for exact bbox lookup.
    """
    index: dict[str, dict] = {}
    for _sheet_name, sheet_data in sheet_cache.items():
        for page_dict in sheet_data.get("data", []):
            if not isinstance(page_dict, dict):
                continue
            for field_name, field_info in page_dict.items():
                if not isinstance(field_info, dict):
                    continue
                existing  = index.get(field_name)
                new_conf  = float(field_info.get("confidence") or 0.0)
                if existing is None or new_conf > float(existing.get("confidence") or 0.0):
                    index[field_name] = {
                        "value":            field_info.get("value", ""),
                        "confidence":       new_conf,
                        "bounding_polygon": field_info.get("bounding_polygon"),
                        "source_page":      field_info.get("source_page", 1),
                        "page_width":       field_info.get("page_width", 8.5),
                        "page_height":      field_info.get("page_height", 11.0),
                    }
    return index


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CLASSIFICATION  (unchanged, short call)
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFICATION_SYSTEM = textwrap.dedent("""
You are a senior insurance document analyst. Classify the document into exactly one of:
  - FNOL        : First Notice of Loss — initial claim intake / notification
  - Legal       : Court documents, complaints, dockets, attorney correspondence
  - Loss Run    : Tabular claims history, TPA loss run, portfolio reports
  - Medical     : Medical records, bills, EOBs, treatment notes, IMEs

Respond ONLY with valid JSON. No preamble.

{
  "classification": "<FNOL|Legal|Loss Run|Medical>",
  "confidence": <0.0–1.0>,
  "reasoning": "<2-3 sentences>",
  "ambiguities": "<mixed signals or empty string>"
}
""").strip()


def classify_document(full_text: str) -> dict:
    result = _llm_call(
        system_prompt=_CLASSIFICATION_SYSTEM,
        user_prompt=f"Classify this document:\n\n{full_text[:3000]}",
        max_tokens=400,
        label="classify",
    )
    if not result:
        return {
            "classification": "Legal",
            "confidence": 0.5,
            "reasoning": "LLM unavailable — defaulted to Legal.",
            "ambiguities": "",
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3a — ENTITIES + SIGNALS  (Call A — the big call)
# ─────────────────────────────────────────────────────────────────────────────

_ENTITIES_SCHEMA = """
Return ONLY valid JSON — no markdown, no preamble.

IMPORTANT:
  • "azure_di_key" must be the EXACT key from the azure_di_fields dict provided
    (copy character-for-character). Set to null if not in that dict.
  • "value" must be the EXACT text from the document (do not paraphrase).
  • "confidence": 0.95+ explicit, 0.70–0.94 implied, <0.70 uncertain.
  • Extract ONLY the fields listed. Omit any not found in the document.
  • DO NOT include source_text if it would make the response very long;
    a short 1-line snippet is fine, or omit it entirely to stay within token budget.

{
  "entities": {
    "<SEMANTIC_LABEL>": {
      "azure_di_key": "<exact Azure DI field name or null>",
      "value":        "<exact value>",
      "source_text":  "<short verbatim snippet, optional>",
      "confidence":   <0.0–1.0>
    }
  },
  "signals": [
    {
      "type":           "<severity|legal_escalation|fraud_indicator|medical_complexity|coverage_issue>",
      "severity_level": "<Highly Severe|High|Moderate|Low>",
      "description":    "<plain-English explanation>",
      "supporting_text":"<verbatim quote, keep short>"
    }
  ]
}
"""

_SUMMARY_SCHEMA = """
Return ONLY valid JSON — no markdown, no preamble.

{
  "summary": "<200-word max factual summary>",
  "type_specific": {
    "<FIELD_NAME>": {
      "azure_di_key": "<exact Azure DI field name or null>",
      "value":        "<exact value>",
      "confidence":   <0.0–1.0>
    }
  },
  "judge": {
    "classification_reasoning": "<why this doc type>",
    "signal_validation":        "<are signals credible?>",
    "data_quality":             "<what is well-extracted vs missing>",
    "recommendations":          "<what a claims handler should do next>"
  }
}
"""


# ── Field lists per doc type ──────────────────────────────────────────────────

_FNOL_ENTITIES = """
Claim Number, Policy Number, Policy Holder Name, Insured Name,
Loss Date, Loss Time, Date Reported, Description of Loss,
Location of Loss, Contact Name, Contact Phone, Contact Email,
Vehicle Make, Vehicle Model, Vehicle Year, VIN,
Claimant Name, Claimant Address, Claimant Phone,
Adjuster Name, Adjuster Phone, Adjuster Email,
Witness Name, Witness Phone, Police Report Number
"""

_FNOL_TYPE_SPECIFIC = """
Severity, Litigation Risk, Fraud Indicator, Coverage Concern,
Estimated Loss Amount, Recommended Next Step
"""

_LEGAL_ENTITIES = """
Case Number, Filing Date, Last Refreshed, Filing Location, Filing Court,
Judge, Category, Practice Area, Matter Type, Status, Case Last Update,
Docket Prepared For, Line of Business, Docket, Circuit, Division,
Cause of Loss, Cause of Action, Case Complaint Summary,
Plaintiff Name, Plaintiff Attorney, Plaintiff Attorney Firm,
Defendant Name, Defendant Attorney, Defendant Attorney Firm,
Insurance Carrier, Policy Number, Coverage Type,
Incident Date, Incident Location, Damages Sought
"""

_LEGAL_TYPE_SPECIFIC = """
Severity, Litigation Stage, Coverage Issue, Estimated Exposure,
Reservation of Rights, Recommended Defense Strategy
"""

_LOSS_RUN_ENTITIES = """
Report Date, Policy Number, Policy Period Start, Policy Period End,
Named Insured, Carrier, TPA Name, Line of Business,
Total Claims Count, Open Claims Count, Closed Claims Count,
Total Incurred, Total Paid, Total Reserve, Total Indemnity Paid,
Total Medical Paid, Total Expense Paid, Largest Claim Amount,
Average Claim Amount, Loss Ratio, Combined Ratio
"""

_LOSS_RUN_TYPE_SPECIFIC = """
Portfolio Severity, Frequency Trend, Litigation Rate,
Large Loss Count, Large Loss Threshold, Recommended Reserve Action
"""

_MEDICAL_ENTITIES = """
Patient Name, Patient DOB, Patient Gender, Patient ID,
Provider Name, Provider NPI, Provider Facility, Provider Address,
Date of Service, Date of Injury, Diagnosis, Primary ICD Code,
Secondary ICD Codes, Procedure Codes, CPT Codes,
Treatment Description, Medications Prescribed,
Billing Amount, Amount Paid, Amount Denied, Adjustment,
Insurance ID, Group Number, Authorization Number,
Attending Physician, Referring Physician, Facility Name
"""

_MEDICAL_TYPE_SPECIFIC = """
Severity, Medical Complexity, Treatment Duration,
Disability Type, MMI Status, Causation Opinion,
Fraud Indicator, Recommended IME
"""

_DOC_TYPE_ENTITIES = {
    "FNOL":     (_FNOL_ENTITIES,     "severity, legal_escalation, fraud_indicator, coverage_issue"),
    "Legal":    (_LEGAL_ENTITIES,    "severity, legal_escalation, fraud_indicator, coverage_issue"),
    "Loss Run": (_LOSS_RUN_ENTITIES, "severity, legal_escalation, fraud_indicator, coverage_issue"),
    "Medical":  (_MEDICAL_ENTITIES,  "severity, medical_complexity, fraud_indicator, coverage_issue"),
}

_DOC_TYPE_TYPE_SPECIFIC = {
    "FNOL":     _FNOL_TYPE_SPECIFIC,
    "Legal":    _LEGAL_TYPE_SPECIFIC,
    "Loss Run": _LOSS_RUN_TYPE_SPECIFIC,
    "Medical":  _MEDICAL_TYPE_SPECIFIC,
}

_DOC_TYPE_ROLES = {
    "FNOL":     "expert FNOL claims intake specialist",
    "Legal":    "legal claims analyst specialising in insurance litigation documents",
    "Loss Run": "TPA loss run analyst specialising in claims portfolio analysis",
    "Medical":  "medical claims analyst specialising in insurance medical documents",
}


def _entities_system(doc_type: str) -> str:
    entity_fields, signal_types = _DOC_TYPE_ENTITIES.get(
        doc_type, _DOC_TYPE_ENTITIES["Legal"]
    )
    role = _DOC_TYPE_ROLES.get(doc_type, "insurance document analyst")
    return textwrap.dedent(f"""
You are a {role}.

Extract ONLY these entity fields (skip any not present in the document):
{entity_fields}

Signal types to detect: {signal_types}

{_ENTITIES_SCHEMA}
""").strip()


def _summary_system(doc_type: str) -> str:
    ts_fields = _DOC_TYPE_TYPE_SPECIFIC.get(doc_type, _LEGAL_TYPE_SPECIFIC)
    role = _DOC_TYPE_ROLES.get(doc_type, "insurance document analyst")
    return textwrap.dedent(f"""
You are a {role}.

For type_specific, extract ONLY these assessment fields (skip any not present):
{ts_fields}

{_SUMMARY_SCHEMA}
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TWO-CALL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_document(
    full_text: str,
    doc_type: str,
    azure_di_fields: dict[str, dict] | None = None,
) -> dict:
    """
    Two-call analysis to avoid token-limit truncation:
      Call A: entities + signals  (larger, needs more tokens)
      Call B: summary + type_specific + judge  (smaller)
    """
    # Build compact name→value map for the LLM prompt (no bbox data)
    adi_kv: dict[str, str] = {}
    if azure_di_fields:
        for fname, fdata in azure_di_fields.items():
            v = fdata.get("value", "")
            if v:
                adi_kv[fname] = str(v)[:200]   # cap long values

    # Truncate text; keep more than before since we split the load
    text_a = full_text[:5000]
    if len(full_text) > 5000:
        text_a += "\n\n[... document truncated ...]"

    text_b = full_text[:3000]
    if len(full_text) > 3000:
        text_b += "\n\n[... document truncated ...]"

    # Azure DI field listing for prompt
    adi_listing = ""
    if adi_kv:
        lines = [f'  "{k}": "{v}"' for k, v in list(adi_kv.items())[:100]]
        adi_listing = (
            "\n\n--- AZURE DOCUMENT INTELLIGENCE FIELDS (use exact key names as azure_di_key) ---\n{\n"
            + ",\n".join(lines)
            + "\n}"
        )

    # ── Call A: entities + signals ────────────────────────────────────────────
    user_a = (
        f"Document type: {doc_type}\n"
        f"Extract entities and detect signals."
        f"{adi_listing}\n\n"
        f"--- DOCUMENT TEXT ---\n{text_a}"
    )
    result_a = _llm_call(
        system_prompt=_entities_system(doc_type),
        user_prompt=user_a,
        max_tokens=3500,
        label="entities_signals",
    )

    # ── Call B: summary + type_specific + judge ───────────────────────────────
    user_b = (
        f"Document type: {doc_type}\n"
        f"Generate a summary and assessment."
        f"{adi_listing}\n\n"
        f"--- DOCUMENT TEXT ---\n{text_b}"
    )
    result_b = _llm_call(
        system_prompt=_summary_system(doc_type),
        user_prompt=user_b,
        max_tokens=1200,
        label="summary_judge",
    )

    # ── Merge results ──────────────────────────────────────────────────────────
    entities      = {}
    signals       = []
    summary       = ""
    type_specific = {}
    judge         = {}

    if result_a:
        entities = result_a.get("entities") or {}
        signals  = result_a.get("signals")  or []
        # Ensure azure_di_key present on every entity
        for _, ed in entities.items():
            if isinstance(ed, dict):
                ed.setdefault("azure_di_key", None)

    if result_b:
        summary       = result_b.get("summary")       or ""
        type_specific = result_b.get("type_specific") or {}
        judge         = result_b.get("judge")         or {}

    if not entities and not signals and not summary:
        return _empty_analysis(doc_type)

    judge.setdefault("classification_reasoning", "")
    judge.setdefault("signal_validation", "")
    judge.setdefault("data_quality", "")
    judge.setdefault("recommendations", "")

    return {
        "summary":       summary,
        "entities":      entities,
        "signals":       signals,
        "type_specific": type_specific,
        "judge":         judge,
    }


def _empty_analysis(doc_type: str) -> dict:
    return {
        "summary": "Analysis unavailable — LLM could not be reached.",
        "entities": {},
        "signals": [],
        "type_specific": {},
        "judge": {
            "classification_reasoning": f"Classified as {doc_type}.",
            "signal_validation": "No signals detected.",
            "data_quality": "LLM unavailable — check OPENAI env vars and token quotas.",
            "recommendations": "Manual review required.",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pdf_intelligence(parsed: dict, sheet_cache: dict | None = None) -> dict:
    """
    Full intelligence pipeline for a parsed PDF.

    Args:
        parsed:       Output of parse_pdf_with_azure().
        sheet_cache:  st.session_state["sheet_cache"] — used to build the Azure
                      DI field index for exact bbox lookup and LLM field hints.

    Returns:
        {
          "full_text":      str,
          "classification": { classification, confidence, reasoning, ambiguities },
          "analysis":       { summary, entities, signals, type_specific, judge },
          "page_count":     int,
          "doc_type":       str,
          "azure_di_index": dict[str, dict],
        }
    """
    full_text  = extract_full_text_from_parsed(parsed)
    page_count = len(parsed.get("pages", []))

    # Build Azure DI index once
    azure_di_index: dict[str, dict] = {}
    if sheet_cache:
        azure_di_index = build_azure_di_field_index(sheet_cache)

    # Classification (short, cheap)
    classification = classify_document(full_text)
    doc_type       = classification.get("classification", "Legal")

    # Two-call analysis
    analysis = analyse_document(full_text, doc_type, azure_di_fields=azure_di_index)

    return {
        "full_text":      full_text,
        "classification": classification,
        "analysis":       analysis,
        "page_count":     page_count,
        "doc_type":       doc_type,
        "azure_di_index": azure_di_index,

    }

    
