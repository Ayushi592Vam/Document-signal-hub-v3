"""
modules/pdf_intelligence.py  — v4

Root causes fixed:
  1. Single LLM call tried to produce summary + entities + signals + type_specific
     + judge in one shot. With max_tokens=2500 the JSON was routinely truncated,
     causing json.loads() to fail silently → _empty_analysis() → empty entities.

  2. bare `except Exception: return None` hid every failure.

  3. On Streamlit Cloud the response for a 25-field Legal doc with
     verbatim source_text snippets easily hits 2500 tokens.

Fixes:
  • Split into TWO cheaper calls:
      Call A — entities + signals  (max_tokens=3500)
      Call B — summary + type_specific + judge  (max_tokens=1000)
  • JSON repair: if json.loads() fails, attempt to close truncated JSON before
    giving up (handles the single most common cloud failure mode).
  • Debug mode: set env var PDF_INTEL_DEBUG=1 to surface raw LLM responses in
    st.session_state["_pdf_intel_debug"] for inspection.
  • entities prompt asks for azure_di_key so UI can do exact bbox lookup.
  • Model routing: standard model for main analysis; enhanced model for validation.
    Model names are never exposed in the UI.
  • FIXED: removed st.write() from module-level code (caused silent crash in
    cloud deployments, resulting in empty entities). All retry logging now goes
    to _debug_store() only.
"""

from __future__ import annotations

import json
import os
import re
import textwrap


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ROUTING  — internal only, never surfaced in the UI
# ─────────────────────────────────────────────────────────────────────────────

def _deployment_standard() -> str:
    """Model used for classification, entities, signals, summary, judge."""
    return os.environ.get("OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")


def _deployment_enhanced() -> str:
    """Model used for the Validation tab — higher reasoning capability."""
    return os.environ.get("OPENAI_DEPLOYMENT_NAME_ENHANCED", "gpt-4o")


# ─────────────────────────────────────────────────────────────────────────────
# AZURE OPENAI CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def _get_openai_client():
    try:
        from openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=os.environ.get("OPENAI_DEPLOYMENT_ENDPOINT", ""),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            api_version=os.environ.get("OPENAI_API_VERSION", "2024-12-01-preview"),
        )
    except Exception:
        return None


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
    use_enhanced: bool = False,
) -> dict | None:
    """
    Call the LLM and return parsed JSON, or None on failure.
    use_enhanced=True routes to the enhanced model (Validation tab only).
    Stores raw response in session state when PDF_INTEL_DEBUG=1.
    Model names are never included in debug keys visible in the UI.
    """
    client = _get_openai_client()
    if not client:
        _debug_store(label, "ERROR: no client (check OPENAI env vars)")
        return None

    model = _deployment_enhanced() if use_enhanced else _deployment_standard()

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        if not raw.strip().endswith("}"):
            _debug_store(label + "_TRUNCATED", raw)
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


def _build_azure_di_index_from_parsed(parsed: dict) -> dict:
    """
    Build azure_di_index directly from parse_pdf_with_azure() output.
    This is ALL pages, not just visited ones.
    """
    index: dict[str, dict] = {}
    for page in parsed.get("pages", []):
        for field in page.get("fields", []):
            fname = (field.get("field_name") or "").strip()
            if not fname:
                continue
            existing = index.get(fname)
            new_conf = float(field.get("confidence", 0.0))
            if existing is None or new_conf > float(existing.get("confidence", 0.0)):
                index[fname] = {
                    "value":            field.get("value", ""),
                    "confidence":       new_conf,
                    "bounding_polygon": field.get("bounding_polygon"),
                    "source_page":      field.get("source_page", page.get("page_num", 1)),
                    "page_width":       field.get("page_width",  8.5),
                    "page_height":      field.get("page_height", 11.0),
                }
    return index


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CLASSIFICATION  (standard model)
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
        use_enhanced=False,
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
# STEP 3a — ENTITIES + SIGNALS  (standard model — Call A)
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

# ── Validation schema — used by the Validation tab (enhanced model) ───────────
_VALIDATION_SCHEMA = """
Return ONLY valid JSON — no markdown, no preamble.

{
  "extraction_accuracy": {
    "score": <0–100>,
    "verdict": "<Pass|Fail|Review>",
    "findings": "<detailed assessment of extraction quality>",
    "missed_fields": ["<field>"],
    "incorrect_fields": [{"field": "<name>", "extracted": "<val>", "expected": "<val>"}]
  },
  "signal_credibility": {
    "score": <0–100>,
    "verdict": "<Credible|Questionable|Unsupported>",
    "findings": "<are detected signals supported by document evidence>",
    "false_positives": ["<signal description>"],
    "missed_signals": ["<signal description>"]
  },
  "coverage_analysis": {
    "score": <0–100>,
    "verdict": "<Adequate|Gaps Identified|Critical Gaps>",
    "findings": "<coverage completeness and any concerns>",
    "gaps": ["<gap description>"]
  },
  "overall_validation": {
    "score": <0–100>,
    "verdict": "<Validated|Needs Review|Failed>",
    "confidence": <0.0–1.0>,
    "summary": "<2-3 sentence overall assessment>",
    "recommended_actions": ["<action>"]
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


def _validation_system(doc_type: str) -> str:
    """System prompt for the Validation tab — uses enhanced model."""
    entity_fields, signal_types = _DOC_TYPE_ENTITIES.get(
        doc_type, _DOC_TYPE_ENTITIES["Legal"]
    )
    role = _DOC_TYPE_ROLES.get(doc_type, "insurance document analyst")
    return textwrap.dedent(f"""
You are a senior {role} performing rigorous quality validation of AI-extracted insurance document data.

Your task is to critically evaluate:
1. EXTRACTION ACCURACY — were the right fields extracted with correct values?
2. SIGNAL CREDIBILITY — are the detected risk signals supported by the document?
3. COVERAGE ANALYSIS — are there gaps, omissions, or coverage concerns missed?
4. OVERALL VALIDATION — holistic assessment with recommended actions.

Expected fields for a {doc_type} document: {entity_fields}
Expected signal types: {signal_types}

Be precise, critical, and actionable. Identify specific errors by field name.
Score each dimension 0-100 (100 = perfect).

{_VALIDATION_SCHEMA}
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TWO-CALL ANALYSIS  (standard model)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_document(
    full_text: str,
    doc_type: str,
    azure_di_fields: dict[str, dict] | None = None,
) -> dict:
    """
    Two-call analysis using the standard model to avoid token-limit truncation:
      Call A: entities + signals  (larger, needs more tokens)
      Call B: summary + type_specific + judge  (smaller)

    NOTE: No st.write() or any Streamlit calls here — this is a pure module.
    All logging goes through _debug_store() only.
    """
    # Build compact name→value map for the LLM prompt (no bbox data)
    adi_kv: dict[str, str] = {}
    if azure_di_fields:
        for fname, fdata in azure_di_fields.items():
            v = fdata.get("value", "")
            if v:
                adi_kv[fname] = str(v)[:200]

    text_a = full_text[:3000]
    if len(full_text) > 3000:
        text_a += "\n\n[... document truncated ...]"

    text_b = full_text[:3000]
    if len(full_text) > 3000:
        text_b += "\n\n[... document truncated ...]"

    # Azure DI field listing for prompt
    adi_listing = ""
    if adi_kv:
        lines = [f'  "{k}": "{v[:50]}"' for k, v in list(adi_kv.items())[:30]]
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
        max_tokens=2500,
        label="entities_signals",
        use_enhanced=False,
    )

    # ── Retry with reduced input if Call A failed (silent — no st.write) ──────
    if result_a is None:
        _debug_store("entities_signals_retry_triggered", "Call A returned None — retrying with reduced input")
        reduced_user_a = user_a[:int(len(user_a) * 0.6)]
        result_a = _llm_call(
            system_prompt=_entities_system(doc_type),
            user_prompt=reduced_user_a,
            max_tokens=2500,
            label="entities_signals_retry",
            use_enhanced=False,
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
        use_enhanced=False,
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
# VALIDATION PIPELINE  (enhanced model — called on demand from UI)
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    full_text: str,
    doc_type: str,
    extracted_entities: dict,
    detected_signals: list,
    azure_di_fields: dict | None = None,
) -> dict:
    """
    Run deep validation of extracted data using the enhanced model.
    Called on demand from the Validation tab — NOT part of the main pipeline.
    Returns a structured validation report dict.
    """
    # Summarise extracted data for the prompt
    entity_summary = json.dumps(
        {k: v.get("value", "") for k, v in extracted_entities.items() if isinstance(v, dict)},
        indent=2,
    )[:2000]

    signal_summary = json.dumps(
        [{"type": s.get("type"), "severity": s.get("severity_level"),
          "description": s.get("description", "")[:100]} for s in detected_signals],
        indent=2,
    )[:800]

    adi_summary = ""
    if azure_di_fields:
        lines = [f'  "{k}": "{str(v.get("value",""))[:60]}"'
                 for k, v in list(azure_di_fields.items())[:25]]
        adi_summary = "\n\nAZURE DI FIELDS:\n{\n" + ",\n".join(lines) + "\n}"

    user_prompt = (
        f"Document type: {doc_type}\n\n"
        f"EXTRACTED ENTITIES:\n{entity_summary}\n\n"
        f"DETECTED SIGNALS:\n{signal_summary}"
        f"{adi_summary}\n\n"
        f"--- DOCUMENT TEXT (truncated) ---\n{full_text[:2500]}"
    )

    result = _llm_call(
        system_prompt=_validation_system(doc_type),
        user_prompt=user_prompt,
        max_tokens=2000,
        label="validation",
        use_enhanced=True,
    )

    if not result:
        return _empty_validation()

    # Ensure all keys present
    result.setdefault("extraction_accuracy", _empty_validation_section("Review"))
    result.setdefault("signal_credibility",  _empty_validation_section("Review"))
    result.setdefault("coverage_analysis",   _empty_validation_section("Review"))
    result.setdefault("overall_validation",  _empty_validation_section("Review"))

    return result


def _empty_validation_section(verdict: str = "Review") -> dict:
    return {
        "score": 0,
        "verdict": verdict,
        "findings": "Validation unavailable.",
    }


def _empty_validation() -> dict:
    return {
        "extraction_accuracy": _empty_validation_section(),
        "signal_credibility":  _empty_validation_section(),
        "coverage_analysis":   _empty_validation_section(),
        "overall_validation": {
            "score": 0,
            "verdict": "Failed",
            "confidence": 0.0,
            "summary": "Validation could not be completed — enhanced AI unavailable.",
            "recommended_actions": ["Check OPENAI environment variables and retry."],
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

    # Build Azure DI index once from all pages
    azure_di_index = _build_azure_di_index_from_parsed(parsed)

    # Classification (short, cheap — standard model)
    classification = classify_document(full_text)
    doc_type       = classification.get("classification", "Legal")

    # Two-call analysis (standard model)
    analysis = analyse_document(full_text, doc_type, azure_di_fields=azure_di_index)

    return {
        "full_text":      full_text,
        "classification": classification,
        "analysis":       analysis,
        "page_count":     page_count,
        "doc_type":       doc_type,
        "azure_di_index": azure_di_index,
    }