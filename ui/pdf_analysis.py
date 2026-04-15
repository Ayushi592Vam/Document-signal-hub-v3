"""
ui/pdf_analysis.py  — v6

Architecture:
  • LLM (pdf_intelligence) extracts ONLY relevant fields per doc type
    (Legal / FNOL / Loss Run / Medical each have a fixed field list)
  • For every LLM-extracted entity, Azure DI sheet_cache is searched
    for a bounding polygon + Azure DI confidence score via name matching
  • The 👁 button shows: zoomed PDF crop + highlight + confidence pill overlay

Tabs (6):
  1. 🔍 Entities (N)          — LLM fields + Azure DI bbox + conf
  2. 📝 Summary               — Classification badge + LLM summary + edit annotations
  3. ⚡ Signals (N)           — Grouped by taxonomy: Highly Severe / High / Moderate / Low
  4. 📄 Raw JSON              — All pages KV JSON (reflects edits), download
  5. 🔄 Transformation Journey — Step-by-step pipeline trace + audit log
  6. 🧑‍⚖️ AI Assessment        — 4 judge cards only
"""

from __future__ import annotations

import datetime
import json
import os
import re

import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_DOC_TYPE_META = {
    "FNOL":     {"icon": "🚨", "color": "#f87171", "bg": "rgba(248,113,113,0.08)"},
    "Legal":    {"icon": "⚖️",  "color": "#a78bfa", "bg": "rgba(167,139,250,0.08)"},
    "Loss Run": {"icon": "📊", "color": "#34d399", "bg": "rgba(52,211,153,0.08)"},
    "Medical":  {"icon": "🏥", "color": "#60a5fa", "bg": "rgba(96,165,250,0.08)"},
}

_SIGNAL_META = {
    "severity":           {"icon": "🔴", "label": "Severity",           "color": "#f87171"},
    "legal_escalation":   {"icon": "⚖️",  "label": "Legal Escalation",   "color": "#a78bfa"},
    "fraud_indicator":    {"icon": "🚩", "label": "Fraud Indicator",    "color": "#fbbf24"},
    "medical_complexity": {"icon": "🏥", "label": "Medical Complexity", "color": "#60a5fa"},
    "coverage_issue":     {"icon": "📋", "label": "Coverage Issue",     "color": "#f59e0b"},
}

_TAXONOMY = {
    "Highly Severe": {"color": "#ff4444", "bg": "rgba(255,68,68,0.10)",   "icon": "🔥"},
    "High":          {"color": "#f87171", "bg": "rgba(248,113,113,0.08)", "icon": "🔴"},
    "Moderate":      {"color": "#f5c842", "bg": "rgba(245,200,66,0.08)",  "icon": "🟡"},
    "Low":           {"color": "#34d399", "bg": "rgba(52,211,153,0.08)",  "icon": "🟢"},
}

_TXT = "#f0efff"
_LBL = "#8888bb"

_UPLOADER_PLUS_CSS = """
<style>
[data-testid="stFileUploaderDropzone"] > div > button:last-of-type,
[data-testid="stFileUploaderDropzone"] button[title="Add files"] {
    display: none !important;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# STYLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _conf_badge(conf: float) -> str:
    pct = int(conf * 100)
    c   = "#34d399" if pct >= 80 else "#f5c842" if pct >= 60 else "#f87171"
    return (
        f"<span style='background:{c}20;border:1px solid {c};border-radius:20px;"
        f"padding:1px 8px;font-size:10px;color:{c};font-weight:700;"
        f"font-family:monospace;'>{pct}%</span>"
    )


def _section_header(title: str, subtitle: str = "") -> str:
    sub = (
        f"<span style='font-size:10px;color:{_LBL};font-family:monospace;'>{subtitle}</span>"
        if subtitle else ""
    )
    return (
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:14px;'>"
        f"<div style='font-size:11px;font-weight:700;color:#c4c4e8;font-family:monospace;"
        f"text-transform:uppercase;letter-spacing:1.5px;white-space:nowrap;'>{title}</div>"
        f"{sub}"
        f"<div style='flex:1;height:1px;background:linear-gradient(90deg,#2a2a45,transparent);'>"
        f"</div></div>"
    )


def _card(content: str, border_color: str = "#2a2a45", bg: str = "#12121c") -> str:
    return (
        f"<div style='background:{bg};border:1px solid {border_color};"
        f"border-radius:8px;padding:14px 16px;margin-bottom:10px;'>{content}</div>"
    )


def _source_snippet(source_text: str) -> str:
    if not source_text:
        return ""
    return (
        f"<div style='font-size:10px;color:{_LBL};font-family:monospace;"
        f"background:#0d0d1a;border-left:2px solid #2a2a45;padding:4px 8px;"
        f"margin-top:5px;border-radius:0 4px 4px 0;font-style:italic;'>"
        f"📄 {source_text}</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# KEY NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _nk(s: str) -> str:
    """Normalise a field name for fuzzy matching."""
    return re.sub(r"[\s\-_:./]+", "_", s.lower()).strip("_")


def _match_score(a: str, b: str) -> float:
    """
    Return a match score 0‒1 between two normalised field-name strings.
    1.0 = exact, >0 = partial overlap, 0 = no match.
    """
    if a == b:
        return 1.0
    # One fully contains the other AND the shorter is ≥ 60% of longer
    shorter = min(len(a), len(b))
    longer  = max(len(a), len(b))
    if longer == 0:
        return 0.0
    if (a in b or b in a) and shorter / longer >= 0.60:
        return shorter / longer
    # Word-level intersection
    a_words = set(a.split("_"))
    b_words = set(b.split("_"))
    # Ignore single-char tokens and very generic words
    _STOP = {"a", "of", "to", "in", "on", "by", "at", "id", "no", "date",
              "name", "type", "code", "the", "and", "or"}
    a_sig = a_words - _STOP - {w for w in a_words if len(w) <= 1}
    b_sig = b_words - _STOP - {w for w in b_words if len(w) <= 1}
    if not a_sig or not b_sig:
        return 0.0
    inter = len(a_sig & b_sig)
    union = len(a_sig | b_sig)
    jaccard = inter / union if union else 0.0
    # Only count word-level matches if at least 1 significant word matches
    # and jaccard is meaningful
    return jaccard if inter >= 1 and jaccard >= 0.40 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# AZURE DI LOOKUP TABLE  (built once per render, cached in session)
# ─────────────────────────────────────────────────────────────────────────────

def _build_azure_lookup() -> dict[str, dict]:
    """
    Build a normalised-name → field_info lookup from ALL Azure DI fields
    currently in sheet_cache. Cached in session state under '_adi_lookup'.
    """
    cache_key = "_adi_lookup"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    lookup: dict[str, dict] = {}
    sheet_cache = st.session_state.get("sheet_cache", {})
    for _, sheet_data in sheet_cache.items():
        for page_dict in sheet_data.get("data", []):
            if not isinstance(page_dict, dict):
                continue
            for az_name, az_info in page_dict.items():
                if not isinstance(az_info, dict):
                    continue
                norm = _nk(az_name)
                # Keep the entry with highest confidence when names collide
                existing = lookup.get(norm)
                new_conf = float(az_info.get("confidence", 0.0))
                if existing is None or new_conf > float(existing.get("confidence", 0.0)):
                    lookup[norm] = az_info

    # Only cache if we found something — avoids caching an empty dict
    # when sheet_cache hasn't been populated yet
    if lookup:
        st.session_state[cache_key] = lookup
    return lookup


def _find_azure_match(entity_name: str, lookup: dict[str, dict]) -> dict | None:
    """
    Find the best-matching Azure DI field for an LLM entity name.
    Returns the az_info dict or None.
    """
    en = _nk(entity_name)
    best_info:  dict | None = None
    best_score: float       = 0.0

    for az_norm, az_info in lookup.items():
        score = _match_score(en, az_norm)
        if score > best_score:
            best_score = score
            best_info  = az_info

    # Require at least a 0.60 score to count as a real match
    return best_info if best_score >= 0.60 else None


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def _lookup_confidence(field_name: str, field_info: dict) -> float:
    """
    Best confidence available:
      1. field_info["confidence"] (set when building intel fields)
      2. Azure DI match confidence
      3. 0.85 if bounding polygon exists (Azure KV confirmed)
      4. 0.0 (shown as N/A)
    """
    direct = field_info.get("confidence")
    if direct is not None and float(direct) > 0:
        return float(direct)
    if field_info.get("bounding_polygon"):
        return 0.85
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LLM ENTITIES ENRICHED WITH AZURE DI  (core function)
# ─────────────────────────────────────────────────────────────────────────────

def _get_intelligence_entities(selected_sheet: str) -> list[tuple[str, dict]]:
    """
    Build the authoritative entity list for the Entities tab.

    Priority order for source of field names:
      1. intelligence["analysis"]["entities"]   — LLM-extracted, type-specific
      2. intelligence["analysis"]["type_specific"] — LLM assessment fields (fallback)
    For each entity, Azure DI bounding polygon + confidence is looked up by name.
    Returns [] only if neither LLM source has data.
    """
    intel    = st.session_state.get("_pdf_intelligence", {})
    analysis = intel.get("analysis", {})
    entities = analysis.get("entities", {})

    # Fallback: use type_specific if entities is empty
    if not entities:
        ts = analysis.get("type_specific", {})
        if ts:
            entities = ts
        else:
            return []

    az_lookup = _build_azure_lookup()
    eds       = _edits()
    out: list[tuple[str, dict]] = []

    for entity_name, entity_data in entities.items():
        if not isinstance(entity_data, dict):
            continue

        llm_value = entity_data.get("value", "")
        llm_conf  = float(entity_data.get("confidence", 0.0))

        # Start with LLM data
        field_info: dict = {
            "value":              llm_value,
            "modified":           eds.get(entity_name, llm_value),
            "confidence":         llm_conf,
            "source_text":        entity_data.get("source_text", ""),
            "source_page":        1,
            "page_width":         8.5,
            "page_height":        11.0,
            "bounding_polygon":   None,
            "_adi_confidence":    0.0,   # Azure DI engine confidence
            "_from_intelligence": True,
        }

        # Look up matching Azure DI field
        az_match = _find_azure_match(entity_name, az_lookup)
        if az_match:
            adi_conf = float(az_match.get("confidence", 0.0))

            # Bounding box from Azure DI
            if az_match.get("bounding_polygon"):
                field_info["bounding_polygon"] = az_match["bounding_polygon"]
                field_info["source_page"]      = az_match.get("source_page", 1)
                field_info["page_width"]        = az_match.get("page_width",  8.5)
                field_info["page_height"]       = az_match.get("page_height", 11.0)

            # Azure DI confidence overrides LLM confidence when available
            if adi_conf > 0:
                field_info["confidence"]     = adi_conf
                field_info["_adi_confidence"] = adi_conf

            # Use Azure DI value only if LLM returned empty
            if not llm_value and az_match.get("value"):
                az_val = az_match["value"]
                field_info["value"]    = az_val
                field_info["modified"] = eds.get(entity_name, az_val)

        out.append((entity_name, field_info))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# SESSION-STATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_raw_fields(selected_sheet: str) -> list[tuple[str, dict]]:
    """Raw Azure DI fields for the current sheet."""
    data = (
        st.session_state
        .get("sheet_cache", {})
        .get(selected_sheet, {})
        .get("data", [])
    )
    seen: set  = set()
    out:  list = []
    for page_dict in data:
        if isinstance(page_dict, dict):
            for fname, finfo in page_dict.items():
                if fname not in seen:
                    seen.add(fname)
                    out.append((fname, finfo))
    return out


def _get_all_pages_fields() -> dict[str, dict[str, str]]:
    """
    {page_label: {field_name: current_value}} for ALL pages in the PDF.
    Uses sheet_cache for visited pages; feature store for unvisited ones.
    Edited values take precedence.
    """
    eds          = _edits()
    cache        = st.session_state.get("sheet_cache", {})
    sheet_names  = st.session_state.get("sheet_names", list(cache.keys()))
    sheet_hashes = st.session_state.get("sheet_hashes", {})
    all_pages: dict[str, dict[str, str]] = {}

    def _extract_kv(data: list) -> dict[str, str]:
        kv: dict[str, str] = {}
        seen: set = set()
        for page_dict in data:
            if isinstance(page_dict, dict):
                for fname, finfo in page_dict.items():
                    if fname not in seen:
                        seen.add(fname)
                        kv[fname] = eds.get(
                            fname,
                            (finfo.get("modified", finfo.get("value", ""))
                             if isinstance(finfo, dict) else str(finfo))
                        )
        return kv

    for sname in sheet_names:
        if sname in cache:
            kv = _extract_kv(cache[sname].get("data", []))
            if kv:
                all_pages[sname] = kv
            continue
        sh_hash = sheet_hashes.get(sname, "")
        if not sh_hash:
            continue
        try:
            from modules.storage import _load_from_feature_store  # type: ignore[import]
            fs = _load_from_feature_store(sh_hash)
            if not fs:
                continue
            kv = {}
            for _cid, rec in fs.get("records", {}).items():
                for fld, fd in rec.items():
                    if fld not in kv and isinstance(fd, dict) and "value" in fd:
                        kv[fld] = eds.get(fld, fd.get("modified", fd.get("value", "")))
            if kv:
                all_pages[sname] = kv
        except Exception:
            pass

    return all_pages


def _edits() -> dict:
    if "_pdf_edits" not in st.session_state:
        st.session_state["_pdf_edits"] = {}
    return st.session_state["_pdf_edits"]


def _edit_history() -> dict:
    if "_pdf_edit_hist" not in st.session_state:
        st.session_state["_pdf_edit_hist"] = {}
    return st.session_state["_pdf_edit_hist"]


def _sync_edit(field_name: str, new_value: str, selected_sheet: str) -> None:
    """Persist an edit to sheet_cache, intelligence entities, _pdf_edits, and history."""
    eds  = _edits()
    hist = _edit_history()
    old  = eds.get(field_name)

    # Update sheet_cache raw field
    data = (
        st.session_state.get("sheet_cache", {})
        .get(selected_sheet, {})
        .get("data", [])
    )
    for page_dict in data:
        if isinstance(page_dict, dict) and field_name in page_dict:
            if old is None:
                old = page_dict[field_name].get("modified", page_dict[field_name].get("value", ""))
            page_dict[field_name]["modified"] = new_value
            break

    # Update intelligence entities
    intel    = st.session_state.get("_pdf_intelligence", {})
    entities = intel.get("analysis", {}).get("entities", {})
    if field_name in entities and isinstance(entities[field_name], dict):
        if old is None:
            old = entities[field_name].get("value", "")

    eds[field_name] = new_value

    if field_name not in hist:
        hist[field_name] = []
    if not hist[field_name] or hist[field_name][-1]["to"] != new_value:
        hist[field_name].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "from":      old or "",
            "to":        new_value,
        })

    # Invalidate the Azure DI lookup cache so it's rebuilt on next render
    st.session_state.pop("_adi_lookup", None)


# ─────────────────────────────────────────────────────────────────────────────
# BOUNDING BOX POPUP
# ─────────────────────────────────────────────────────────────────────────────

def _render_bbox_content(field_name: str, field_info: dict, pdf_path: str) -> None:
    bounding_polygon = field_info.get("bounding_polygon")
    source_page      = int(field_info.get("source_page", 1))
    page_width       = float(field_info.get("page_width",  8.5))
    page_height      = float(field_info.get("page_height", 11.0))
    extracted_value  = field_info.get("value", "")
    confidence       = _lookup_confidence(field_name, field_info)
    conf_pct         = int(confidence * 100)
    conf_hex         = "#34d399" if conf_pct >= 80 else "#f5c842" if conf_pct >= 60 else "#f87171"
    conf_rgb         = (
        (0.20, 0.83, 0.60) if conf_pct >= 80 else
        (0.96, 0.78, 0.26) if conf_pct >= 60 else
        (0.97, 0.44, 0.44)
    )

    # Info header card
    st.markdown(
        f"<div style='background:#12121c;border:1px solid #2a2a45;border-radius:8px;"
        f"padding:14px 16px;margin-bottom:14px;'>"
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;'>"
        f"<div>"
        f"<div style='font-size:9px;color:{_LBL};font-family:monospace;"
        f"text-transform:uppercase;letter-spacing:1px;'>LLM-Extracted Field</div>"
        f"<div style='font-size:16px;font-weight:700;color:#a78bfa;"
        f"font-family:monospace;margin-top:2px;'>{field_name}</div>"
        f"</div>"
        f"<div style='text-align:right;'>"
        f"<div style='font-size:9px;color:{_LBL};font-family:monospace;"
        f"text-transform:uppercase;letter-spacing:1px;'>Azure DI Confidence</div>"
        f"<div style='font-size:28px;font-weight:800;color:{conf_hex};"
        f"font-family:monospace;margin-top:2px;'>"
        f"{'N/A' if conf_pct == 0 else f'{conf_pct}%'}</div>"
        f"</div></div>"
        f"<div style='height:1px;background:#1e1e30;margin:10px 0;'></div>"
        f"<div style='font-size:9px;color:{_LBL};font-family:monospace;"
        f"text-transform:uppercase;letter-spacing:1px;'>Extracted Value</div>"
        f"<div style='font-size:13px;color:{_TXT};font-family:monospace;"
        f"background:#0d0d1a;padding:7px 10px;border-radius:4px;margin-top:4px;"
        f"word-break:break-word;'>{extracted_value or '—'}</div>"
        f"<div style='margin-top:8px;font-size:10px;color:#555;font-family:monospace;'>"
        f"Source: Page {source_page} &nbsp;·&nbsp; "
        f"Bounding box: {'✓ available' if bounding_polygon else '✗ not available'}"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    if not bounding_polygon:
        st.warning(
            "⚠ No bounding-box coordinates for this field.\n\n"
            "Azure DI did not return a precise region for this key-value pair. "
            "This typically means the field was extracted from text layout rather "
            "than a structured key-value block."
        )
        return

    if not pdf_path or not os.path.exists(pdf_path):
        st.error("❌ PDF file not accessible for rendering.")
        return

    try:
        import fitz

        doc         = fitz.open(pdf_path)
        total_pages = len(doc)

        if source_page < 1 or source_page > total_pages:
            st.error(f"Page {source_page} out of range ({total_pages} total).")
            doc.close()
            return

        page   = doc[source_page - 1]
        pw_pts = page.rect.width
        ph_pts = page.rect.height

        # Convert inch coordinates → PDF points
        sx  = pw_pts / page_width
        sy  = ph_pts / page_height
        pts = [(x * sx, y * sy) for x, y in bounding_polygon]
        xs  = [p[0] for p in pts]
        ys  = [p[1] for p in pts]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        bbox = fitz.Rect(x0, y0, x1, y1)

        # ── Coloured highlight rectangle ──────────────────────────────────────
        shape = page.new_shape()
        shape.draw_rect(bbox)
        shape.finish(
            color=conf_rgb,
            fill=(*conf_rgb, 0.22),
            fill_opacity=0.28,
            width=2.5,
        )
        shape.commit()

        # ── Confidence pill drawn above the bounding box ──────────────────────
        if conf_pct > 0:
            label  = f"  {conf_pct}% confidence  "
            char_w = 5.6                              # estimated pts per char at font=8
            pill_w = len(label) * char_w
            ly     = max(y0 - 14, 10)                 # y of pill bottom edge
            lrect  = fitz.Rect(x0, ly - 13, x0 + pill_w, ly + 2)

            pill = page.new_shape()
            pill.draw_rect(lrect)
            pill.finish(color=conf_rgb, fill=conf_rgb, fill_opacity=0.93, width=0)
            pill.commit()

            page.insert_text(
                fitz.Point(x0 + 3, ly - 1),
                f"{conf_pct}% confidence",
                fontsize=8,
                color=(0.04, 0.04, 0.04),
            )

        # ── Zoomed crop ───────────────────────────────────────────────────────
        PAD  = 60
        crop = fitz.Rect(
            max(0,      x0 - PAD),
            max(0,      y0 - PAD - 22),
            min(pw_pts, x1 + PAD),
            min(ph_pts, y1 + PAD),
        )
        pix_zoom = page.get_pixmap(matrix=fitz.Matrix(2.8, 2.8), clip=crop)

        st.markdown(
            "<div style='font-size:11px;font-weight:700;color:#c4c4e8;"
            "font-family:monospace;text-transform:uppercase;letter-spacing:1.5px;"
            "margin-bottom:8px;'>🔍 Zoomed View</div>",
            unsafe_allow_html=True,
        )
        st.image(pix_zoom.tobytes("png"), use_container_width=True)

        with st.expander("📄 Full Page View"):
            pix_full = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            st.image(pix_full.tobytes("png"), use_container_width=True)

        doc.close()

    except ImportError:
        st.error("**PyMuPDF required.** Install: `pip install pymupdf`")
    except Exception as exc:
        st.error(f"Could not render PDF page: {exc}")


_HAS_DIALOG = hasattr(st, "dialog")

if _HAS_DIALOG:
    @st.dialog("📍 Field Location in Document", width="large")
    def _bbox_popup(field_name: str, field_info: dict, pdf_path: str) -> None:
        _render_bbox_content(field_name, field_info, pdf_path)
else:
    def _bbox_popup(field_name: str, field_info: dict, pdf_path: str) -> None:  # type: ignore[misc]
        with st.expander(f"📍 {field_name}", expanded=True):
            _render_bbox_content(field_name, field_info, pdf_path)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — ENTITIES
# ─────────────────────────────────────────────────────────────────────────────

def _render_entities_tab(
    intelligence: dict,
    selected_sheet: str,
    pdf_path: str | None,
) -> None:
    intel_fields = _get_intelligence_entities(selected_sheet)
    eds          = _edits()

    # ── Diagnostics + fallback if LLM entities are empty ────────────────────
    if not intel_fields:
        intel    = st.session_state.get("_pdf_intelligence", {})
        analysis = intel.get("analysis", {})

        # Show diagnostic pill row
        has_intel  = bool(intel)
        has_summ   = bool(analysis.get("summary", "").strip())
        has_ents   = bool(analysis.get("entities"))
        has_ts     = bool(analysis.get("type_specific"))
        has_sigs   = bool(analysis.get("signals"))
        doc_type   = intel.get("doc_type", "")

        def _pill(label: str, ok: bool) -> str:
            c = "#34d399" if ok else "#f87171"
            return (f"<span style='background:{c}18;border:1px solid {c}55;"
                    f"border-radius:20px;padding:3px 10px;font-size:10px;"
                    f"color:{c};font-family:monospace;'>"
                    f"{'✓' if ok else '✗'} {label}</span>")

        st.markdown(
            f"<div style='background:#12121c;border:1px solid #2a2a45;"
            f"border-radius:8px;padding:14px 16px;margin-bottom:12px;'>"
            f"<div style='font-size:11px;font-weight:700;color:#f5c842;"
            f"font-family:monospace;margin-bottom:10px;'>"
            f"⚠ LLM entity extraction returned 0 fields</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;'>"
            f"{_pill('Intelligence ran', has_intel)}"
            f"{_pill('Summary', has_summ)}"
            f"{_pill('Entities', has_ents)}"
            f"{_pill('Type fields', has_ts)}"
            f"{_pill('Signals', has_sigs)}"
            f"{_pill('Doc type: '+doc_type if doc_type else 'Doc type', bool(doc_type))}"
            f"</div>"
            f"<div style='font-size:11px;color:{_LBL};font-family:monospace;'>"
            f"{'The LLM ran but returned no entities — this can happen when the document text is too short, the LLM rate-limited, or the field list did not match document content.' if has_intel else 'Intelligence pipeline has not run yet for this file.'}"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # Re-run button
        col_btn, col_sp = st.columns([2, 5])
        with col_btn:
            if st.button("🔄 Re-run AI Analysis", use_container_width=True,
                         key="_rerun_intelligence_btn"):
                st.session_state.pop("_pdf_intelligence", None)
                st.session_state.pop("_pdf_intelligence_file", None)
                st.session_state.pop("_adi_lookup", None)
                st.rerun()

        # Fall back to raw Azure DI fields with bbox support
        raw = _get_raw_fields(selected_sheet)
        if not raw:
            st.info("No fields extracted for this page.")
            return

        st.markdown(
            f"<div style='font-size:11px;color:{_LBL};font-family:monospace;"
            f"margin:8px 0 12px 0;'>📋 Showing raw Azure Document Intelligence "
            f"fields — bounding boxes and confidence scores are available via 👁</div>",
            unsafe_allow_html=True,
        )
        intel_fields = raw

    # Count how many have bounding boxes
    bbox_count = sum(1 for _, fi in intel_fields if fi.get("bounding_polygon"))

    st.markdown(
        _section_header(
            "Extracted Entities",
            f"{len(intel_fields)} LLM-relevant field(s) · "
            f"{bbox_count} with Azure DI bounding box",
        ),
        unsafe_allow_html=True,
    )

    _HDR = (
        "font-size:10px;font-weight:700;font-family:monospace;"
        "text-transform:uppercase;letter-spacing:1.5px;"
        "padding:6px 4px;border-bottom:1px solid #2a2a45;"
    )
    h1, h2, h3, h4 = st.columns([2.5, 3.5, 3.5, 1.0])
    h1.markdown(f"<div style='{_HDR}color:{_LBL};'>Field Name</div>",                unsafe_allow_html=True)
    h2.markdown(f"<div style='{_HDR}color:#34d399;'>Extracted</div>",                 unsafe_allow_html=True)
    h3.markdown(f"<div style='{_HDR}color:#4f9cf9;'>Modified</div>",                  unsafe_allow_html=True)
    h4.markdown(f"<div style='{_HDR}color:{_LBL};text-align:center;'>Actions</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

    _EM_KEY = "_pdf_edit_mode_fields"
    if _EM_KEY not in st.session_state:
        st.session_state[_EM_KEY] = set()

    _bbox_pending_name: str | None  = None
    _bbox_pending_info: dict | None = None

    for field_name, field_info in intel_fields:
        extracted  = field_info.get("value", "")
        modified   = eds.get(field_name, field_info.get("modified", extracted))
        in_edit    = field_name in st.session_state[_EM_KEY]
        has_bbox   = bool(field_info.get("bounding_polygon"))
        is_changed = modified != extracted
        confidence = _lookup_confidence(field_name, field_info)
        conf_pct   = int(confidence * 100)

        c1, c2, c3, c4 = st.columns([2.5, 3.5, 3.5, 1.0])

        with c1:
            st.markdown(
                f"<div style='font-size:12px;font-weight:600;color:{_TXT};"
                f"font-family:monospace;padding:6px 4px 2px 4px;line-height:1.4;"
                f"word-break:break-word;'>{field_name}</div>"
                + (f"<div style='padding:0 4px 6px 4px;'>{_conf_badge(confidence)}</div>"
                   if conf_pct > 0 else ""),
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"<div style='font-size:12px;color:{_TXT};font-family:monospace;"
                f"background:#0d0d1a;border:1px solid #1e1e30;"
                f"padding:7px 10px;border-radius:5px;min-height:34px;"
                f"line-height:1.5;white-space:pre-wrap;word-break:break-word;'>"
                f"{extracted if extracted else f'<span style=\"color:#3a3a55;\">—</span>'}"
                f"</div>",
                unsafe_allow_html=True,
            )

        with c3:
            if in_edit:
                st.text_input(
                    "modified_value", value=modified,
                    key=f"_pmv_{field_name}", label_visibility="collapsed",
                )
            else:
                _badge = (
                    f"<span style='margin-left:6px;font-size:9px;color:#4f9cf9;"
                    f"border:1px solid #4f9cf9;border-radius:10px;padding:1px 5px;"
                    f"white-space:nowrap;'>✏ edited</span>"
                    if is_changed else ""
                )
                _css = (
                    f"color:{_TXT};background:#0d1a2d;border:1px solid #1e3a5f;"
                    if is_changed else
                    f"color:{_TXT};background:#0d0d1a;border:1px solid #1e1e30;"
                )
                st.markdown(
                    f"<div style='font-size:12px;font-family:monospace;{_css}"
                    f"padding:7px 10px;border-radius:5px;min-height:34px;"
                    f"line-height:1.5;white-space:pre-wrap;word-break:break-word;'>"
                    f"{modified if modified else f'<span style=\"color:#3a3a55;\">—</span>'}"
                    f"{_badge}</div>",
                    unsafe_allow_html=True,
                )

        with c4:
            be, beye = st.columns(2)
            with be:
                lbl = "💾" if in_edit else "✏️"
                if st.button(lbl, key=f"_pbtn_edit_{field_name}",
                             help="Save" if in_edit else "Edit",
                             use_container_width=True):
                    if in_edit:
                        saved = st.session_state.get(f"_pmv_{field_name}", modified)
                        _sync_edit(field_name, saved, selected_sheet)
                        st.session_state[_EM_KEY].discard(field_name)
                    else:
                        st.session_state[_EM_KEY].add(field_name)
                    st.rerun()

            with beye:
                if has_bbox:
                    tip = f"View in document · Azure DI confidence: {conf_pct}%"
                else:
                    tip = "No Azure DI bounding box available for this field"
                if st.button("👁", key=f"_pbtn_eye_{field_name}", help=tip,
                             disabled=not has_bbox, use_container_width=True):
                    _bbox_pending_name = field_name
                    _bbox_pending_info = field_info

        st.markdown(
            "<div style='height:1px;background:#1a1a2e;margin:2px 0 4px 0;'></div>",
            unsafe_allow_html=True,
        )

    if _bbox_pending_name and _bbox_pending_info is not None:
        _bbox_popup(_bbox_pending_name, _bbox_pending_info, pdf_path or "")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_intelligence_kv(selected_sheet: str) -> dict[str, str]:
    """
    Return {field_name: current_value} for all LLM-extracted intelligence entities.
    Uses modified (edited) values where available. Falls back to raw Azure DI fields.
    """
    intel_fields = _get_intelligence_entities(selected_sheet)
    eds          = _edits()

    if not intel_fields:
        intel_fields = _get_raw_fields(selected_sheet)

    kv: dict[str, str] = {}
    for fname, finfo in intel_fields:
        kv[fname] = eds.get(fname, finfo.get("modified", finfo.get("value", "")))
    return kv


def _regenerate_summary(intelligence: dict, selected_sheet: str) -> str | None:
    """
    Call the LLM to regenerate a summary using the CURRENT (possibly edited)
    field values. Returns the new summary string, or None on failure.
    """
    doc_type = intelligence.get("doc_type", "Legal")
    full_text = intelligence.get("full_text", "")

    # Build current state of all fields (modified values take precedence)
    current_kv = _get_intelligence_kv(selected_sheet)
    eds        = _edits()
    hist       = _edit_history()

    # Build a concise "current fields" block for the prompt
    field_lines = []
    for fname, val in current_kv.items():
        orig = ""
        if fname in hist and hist[fname]:
            orig = hist[fname][0].get("from", "")
        if orig and orig != val:
            field_lines.append(f"  {fname}: {val}  [was: {orig}]")
        else:
            field_lines.append(f"  {fname}: {val}")

    fields_block = "\n".join(field_lines) if field_lines else "(no fields extracted)"

    system_prompt = (
        f"You are a senior insurance document analyst. "
        f"You will be given the current (potentially user-edited) field values "
        f"extracted from a {doc_type} insurance document, along with the original "
        f"document text. Generate a concise factual summary (max 200 words) that "
        f"reflects the CURRENT field values — use the edited values, not the originals "
        f"where they differ. Do not include field names in the summary; write natural prose. "
        f"Return ONLY the summary text with no preamble."
    )

    user_prompt = (
        f"Document type: {doc_type}\n\n"
        f"CURRENT FIELD VALUES (use these — some may have been edited by the user):\n"
        f"{fields_block}\n\n"
        f"ORIGINAL DOCUMENT TEXT (for context only — defer to field values above):\n"
        f"{full_text[:4000]}"
        + ("\n[... truncated ...]" if len(full_text) > 4000 else "")
    )

    try:
        import os, re
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=os.environ.get("OPENAI_DEPLOYMENT_ENDPOINT", ""),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            api_version=os.environ.get("OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        deployment = os.environ.get("OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
        response   = client.chat.completions.create(
            model=deployment,
            max_tokens=400,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = re.sub(r"^```.*?```$", "", raw, flags=re.DOTALL).strip()
        return raw if raw else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _render_summary_tab(intelligence: dict, selected_sheet: str) -> None:
    doc_type = intelligence.get("doc_type", "Legal")
    clf      = intelligence.get("classification", {})
    meta     = _DOC_TYPE_META.get(doc_type, _DOC_TYPE_META["Legal"])
    conf     = clf.get("confidence", 0.5)

    # Use regenerated summary if available, else original LLM summary
    _SUMM_KEY = "_pdf_summary_override"
    summary   = st.session_state.get(_SUMM_KEY) or intelligence.get("analysis", {}).get("summary", "")

    st.markdown(
        f"<div style='background:{meta['bg']};border:1px solid {meta['color']}40;"
        f"border-left:4px solid {meta['color']};border-radius:8px;"
        f"padding:14px 18px;margin-bottom:16px;'>"
        f"<div style='display:flex;align-items:center;gap:12px;'>"
        f"<span style='font-size:28px;'>{meta['icon']}</span>"
        f"<div>"
        f"<div style='font-size:20px;font-weight:800;color:{meta['color']};"
        f"font-family:monospace;text-transform:uppercase;letter-spacing:2px;'>{doc_type}</div>"
        f"<div style='font-size:11px;color:{_LBL};margin-top:2px;'>"
        f"Classification confidence: {_conf_badge(conf)}</div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    # ── Regenerate button row ──────────────────────────────────────────────────
    eds  = _edits()
    hist = _edit_history()
    changed = [(fn, h) for fn, h in hist.items() if h]
    is_regenerated = bool(st.session_state.get(_SUMM_KEY))

    btn_col, status_col = st.columns([2, 6])
    with btn_col:
        regen_label = "🔄 Re-regenerate Summary" if is_regenerated else "🔄 Regenerate with Edits"
        regen_help  = (
            "Re-generate the summary using your edited field values. "
            "The LLM will use the modified values as ground truth."
        )
        regen_disabled = not changed
        if st.button(
            regen_label,
            key="_regen_summary_btn",
            help=regen_help if not regen_disabled else "Make edits in the Entities tab first",
            disabled=regen_disabled,
            use_container_width=True,
        ):
            with st.spinner("🧠 Regenerating summary with edited values…"):
                new_summary = _regenerate_summary(intelligence, selected_sheet)
            if new_summary:
                st.session_state[_SUMM_KEY] = new_summary
                summary = new_summary
                st.toast("✅ Summary regenerated with your edits!")
                st.rerun()
            else:
                st.error("Could not regenerate summary — LLM unavailable.")

    with status_col:
        if is_regenerated:
            st.markdown(
                f"<div style='font-size:11px;color:#34d399;font-family:monospace;"
                f"padding-top:8px;'>✓ Showing regenerated summary · based on edited values</div>",
                unsafe_allow_html=True,
            )
        elif changed:
            st.markdown(
                f"<div style='font-size:11px;color:#f5c842;font-family:monospace;"
                f"padding-top:8px;'>⚠ {len(changed)} field(s) edited — click Regenerate to update summary</div>",
                unsafe_allow_html=True,
            )

    if st.session_state.get(_SUMM_KEY):
        if st.button("↩ Reset to original summary", key="_reset_summary_btn",
                     help="Discard regenerated summary and show the original"):
            st.session_state.pop(_SUMM_KEY, None)
            st.rerun()

    st.markdown(_section_header("Document Summary"), unsafe_allow_html=True)

    if summary:
        # Inline-annotate old values → new values only in the original summary
        # (the regenerated summary already has the correct values baked in)
        annotated = summary
        if not is_regenerated:
            for fname, new_val in eds.items():
                changes = hist.get(fname, [])
                if not changes:
                    continue
                old_val = changes[0].get("from", "")
                if old_val and old_val != new_val and old_val in annotated:
                    annotated = annotated.replace(
                        old_val,
                        f"<span style='background:#1e3a5f;color:#4f9cf9;"
                        f"border-radius:3px;padding:0 3px;font-weight:600;"
                        f"border-bottom:2px solid #4f9cf9;'"
                        f"title='Edited from: {old_val}'>{new_val}</span>",
                        1,
                    )

        border_color = "#34d399" if is_regenerated else meta["color"]
        label_html   = (
            f"<div style='font-size:9px;font-weight:700;color:#34d399;"
            f"font-family:monospace;text-transform:uppercase;letter-spacing:1px;"
            f"margin-bottom:6px;'>✓ Regenerated summary — uses your edited values</div>"
            if is_regenerated else ""
        )
        st.markdown(
            f"<div style='background:#0d0d1a;border:1px solid {border_color}30;"
            f"border-radius:8px;padding:16px 20px;font-size:13px;color:{_TXT};"
            f"line-height:1.9;'>{label_html}{annotated}</div>",
            unsafe_allow_html=True,
        )

        # Show edited field diff table
        if changed and not is_regenerated:
            rows_html = ""
            for fname, fchanges in changed:
                old_v = fchanges[0].get("from", "—")
                new_v = eds.get(fname, fchanges[-1].get("to", "—"))
                rows_html += (
                    f"<div style='display:grid;grid-template-columns:180px 1fr auto 1fr;"
                    f"gap:8px;padding:6px 0;border-bottom:1px solid #1a1a2e;align-items:center;'>"
                    f"<span style='font-size:11px;font-weight:600;color:{_TXT};"
                    f"font-family:monospace;'>{fname}</span>"
                    f"<span style='font-size:11px;color:{_LBL};font-family:monospace;"
                    f"text-decoration:line-through;word-break:break-word;'>{old_v}</span>"
                    f"<span style='font-size:13px;color:{_LBL};'>→</span>"
                    f"<span style='font-size:11px;color:#4f9cf9;font-family:monospace;"
                    f"font-weight:600;word-break:break-word;'>{new_v}</span>"
                    f"</div>"
                )
            st.markdown(
                f"<div style='background:#0d1a2d;border:1px solid #1e3a5f;"
                f"border-radius:8px;padding:12px 16px;margin-top:12px;'>"
                f"<div style='font-size:10px;font-weight:700;color:#f5c842;"
                f"font-family:monospace;text-transform:uppercase;letter-spacing:1px;"
                f"margin-bottom:6px;'>⚠ {len(changed)} Pending Edit(s)</div>"
                f"<div style='font-size:9px;color:{_LBL};font-family:monospace;"
                f"margin-bottom:8px;'>These edits are not yet in the summary above. "
                f"Click \"Regenerate with Edits\" to update it.</div>"
                f"{rows_html}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No summary generated.")

    full_text  = intelligence.get("full_text", "")
    page_count = intelligence.get("page_count", 0)
    st.markdown(
        f"<div style='display:flex;gap:14px;margin-top:14px;flex-wrap:wrap;'>"
        + "".join(
            f"<div style='background:#17172a;border:1px solid #2a2a45;"
            f"border-radius:6px;padding:8px 14px;'>"
            f"<div style='font-size:9px;color:#555;font-family:monospace;"
            f"text-transform:uppercase;letter-spacing:1px;'>{lbl}</div>"
            f"<div style='font-size:14px;font-weight:700;color:#4f9cf9;"
            f"font-family:monospace;margin-top:2px;'>{val}</div></div>"
            for lbl, val in [
                ("Pages", page_count),
                ("Words", len(full_text.split())),
                ("Characters", len(full_text)),
                ("Doc Type", doc_type),
            ]
        )
        + "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SIGNALS  (severity taxonomy)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_severity(sig: dict) -> str:
    # LLM-assigned level (preferred)
    llm = (sig.get("severity_level") or "").strip().title()
    if llm in _TAXONOMY:
        return llm

    stype = sig.get("type", "")
    desc  = (sig.get("description", "") + " " + sig.get("supporting_text", "")).lower()

    hs = ["fatal", "death", "fatality", "catastrophic", "permanent disab",
          "punitive", "class action", "multi-party", "major loss"]
    if stype == "severity" and any(k in desc for k in hs):
        return "Highly Severe"
    if stype in ("severity", "legal_escalation"):
        return "High"
    if stype in ("coverage_issue", "medical_complexity"):
        return "Moderate"
    if stype == "fraud_indicator":
        if any(k in desc for k in ["confirmed", "proven", "definite"]):
            return "High"
        return "Moderate"
    return "Low"


def _render_signals_tab(intelligence: dict) -> None:
    signals = intelligence.get("analysis", {}).get("signals", [])

    st.markdown(
        _section_header("Signal Detection", f"{len(signals)} signal(s) detected"),
        unsafe_allow_html=True,
    )

    if not signals:
        st.markdown(
            _card(
                f"<div style='color:#34d399;font-size:13px;font-family:monospace;'>"
                f"✓ No significant signals detected.</div>",
                border_color="#34d39940", bg="#0a1f14",
            ),
            unsafe_allow_html=True,
        )
        return

    grouped: dict[str, list[dict]] = {lv: [] for lv in _TAXONOMY}
    for sig in signals:
        grouped[_classify_severity(sig)].append(sig)

    # Summary pills
    pills = "".join(
        f"<span style='background:{_TAXONOMY[lv]['color']}18;"
        f"border:1px solid {_TAXONOMY[lv]['color']}55;border-radius:20px;"
        f"padding:4px 12px;font-size:11px;font-weight:700;"
        f"color:{_TAXONOMY[lv]['color']};font-family:monospace;'>"
        f"{_TAXONOMY[lv]['icon']} {lv} ({len(sigs)})</span>"
        for lv, sigs in grouped.items()
        if sigs
    )
    if pills:
        st.markdown(
            f"<div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px;'>"
            f"{pills}</div>",
            unsafe_allow_html=True,
        )

    for level in ["Highly Severe", "High", "Moderate", "Low"]:
        group_sigs = grouped.get(level, [])
        if not group_sigs:
            continue
        tax = _TAXONOMY[level]
        tc  = tax["color"]

        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:16px 0 8px 0;'>"
            f"<span style='font-size:16px;'>{tax['icon']}</span>"
            f"<span style='font-size:12px;font-weight:700;color:{tc};"
            f"font-family:monospace;text-transform:uppercase;letter-spacing:1.2px;'>{level}</span>"
            f"<div style='flex:1;height:1px;background:{tc}30;'></div>"
            f"<span style='font-size:10px;color:{tc};font-family:monospace;"
            f"background:{tc}14;border:1px solid {tc}40;border-radius:10px;"
            f"padding:1px 8px;'>{len(group_sigs)} signal(s)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        for sig in group_sigs:
            sig_type = sig.get("type", "unknown")
            m = _SIGNAL_META.get(sig_type, {"icon": "⚠", "label": sig_type, "color": tc})
            c = m["color"]
            st.markdown(
                f"<div style='background:#0d0d1a;border:1px solid {c}35;"
                f"border-left:4px solid {tc};border-radius:8px;"
                f"padding:12px 16px;margin-bottom:8px;'>"
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>"
                f"<span style='font-size:14px;'>{m['icon']}</span>"
                f"<span style='font-size:11px;font-weight:700;color:{c};"
                f"font-family:monospace;text-transform:uppercase;letter-spacing:1px;'>"
                f"{m['label']}</span>"
                f"<span style='margin-left:auto;font-size:9px;color:{tc};"
                f"background:{tc}18;border:1px solid {tc}40;border-radius:10px;"
                f"padding:1px 7px;font-family:monospace;white-space:nowrap;'>"
                f"{tax['icon']} {level}</span>"
                f"</div>"
                f"<div style='font-size:13px;color:{_TXT};line-height:1.7;margin-bottom:6px;'>"
                f"{sig.get('description', '')}</div>"
                + (
                    f"<div style='font-size:11px;color:{_LBL};font-family:monospace;"
                    f"background:#17172a;border-left:2px solid {c}60;padding:5px 10px;"
                    f"border-radius:0 4px 4px 0;font-style:italic;'>"
                    f"📄 \"{sig.get('supporting_text', '')}\"</div>"
                    if sig.get("supporting_text") else ""
                )
                + "</div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — RAW JSON  (LLM-extracted fields, modifications applied)
# ─────────────────────────────────────────────────────────────────────────────

def _render_raw_json_tab(intelligence: dict, selected_sheet: str) -> None:
    """
    Show the LLM-extracted key-value pairs with modifications applied.
    Source: intelligence entities (type-specific fields only), NOT raw Azure DI dump.
    Modified values take precedence over originals.
    """
    eds  = _edits()
    hist = _edit_history()

    # ── Build the primary KV dict from LLM intelligence entities ─────────────
    intel_kv = _get_intelligence_kv(selected_sheet)

    st.markdown(
        _section_header(
            "Extracted Key-Value Pairs",
            f"{len(intel_kv)} LLM-extracted fields · modifications applied",
        ),
        unsafe_allow_html=True,
    )

    if not intel_kv:
        st.info("No LLM-extracted fields available yet. Run AI analysis first.")
        return

    # Count how many fields have been edited
    edited_count = sum(1 for fn in intel_kv if fn in eds and eds[fn] != (
        next((fi.get("value","") for nm, fi in _get_intelligence_entities(selected_sheet)
              if nm == fn), "")
    ))

    # Status banner
    if edited_count:
        st.markdown(
            f"<div style='background:#0d1a2d;border:1px solid #1e3a5f;"
            f"border-radius:6px;padding:8px 14px;margin-bottom:12px;"
            f"font-size:11px;font-family:monospace;color:#4f9cf9;'>"
            f"✏ {edited_count} field(s) show modified values below</div>",
            unsafe_allow_html=True,
        )

    # ── JSON preview with visual diff for edited fields ───────────────────────
    # Build annotated preview (code block shows clean JSON)
    st.code(json.dumps(intel_kv, indent=2, ensure_ascii=False), language="json")

    # ── Diff table: which fields changed ──────────────────────────────────────
    changed = [(fn, h) for fn, h in hist.items() if h and fn in intel_kv]
    if changed:
        rows_html = ""
        for fname, fchanges in changed:
            orig    = fchanges[0].get("from", "—")
            current = eds.get(fname, fchanges[-1].get("to", "—"))
            rows_html += (
                f"<div style='display:grid;grid-template-columns:180px 1fr auto 1fr;"
                f"gap:8px;padding:6px 0;border-bottom:1px solid #1a1a2e;align-items:center;'>"
                f"<span style='font-size:11px;font-weight:600;color:{_TXT};"
                f"font-family:monospace;'>{fname}</span>"
                f"<span style='font-size:11px;color:{_LBL};font-family:monospace;"
                f"text-decoration:line-through;word-break:break-word;'>{orig}</span>"
                f"<span style='font-size:13px;color:{_LBL};'>→</span>"
                f"<span style='font-size:11px;color:#34d399;font-family:monospace;"
                f"font-weight:600;word-break:break-word;'>{current}</span>"
                f"</div>"
            )
        with st.expander(f"📋 {len(changed)} modified field(s) — click to see diff"):
            st.markdown(
                f"<div style='background:#0d0d1a;border:1px solid #2a2a45;"
                f"border-radius:8px;padding:12px 16px;'>"
                f"<div style='display:grid;grid-template-columns:180px 1fr auto 1fr;"
                f"gap:8px;padding-bottom:6px;border-bottom:1px solid #2a2a45;margin-bottom:4px;'>"
                f"<span style='font-size:9px;color:{_LBL};font-family:monospace;"
                f"text-transform:uppercase;letter-spacing:1px;'>Field</span>"
                f"<span style='font-size:9px;color:#f87171;font-family:monospace;"
                f"text-transform:uppercase;letter-spacing:1px;'>Original</span>"
                f"<span></span>"
                f"<span style='font-size:9px;color:#34d399;font-family:monospace;"
                f"text-transform:uppercase;letter-spacing:1px;'>Modified</span>"
                f"</div>{rows_html}</div>",
                unsafe_allow_html=True,
            )

    # ── Download buttons ──────────────────────────────────────────────────────
    full_json_str = json.dumps(intel_kv, indent=2, ensure_ascii=False)
    st.markdown(
        f"<div style='font-size:11px;color:{_LBL};font-family:monospace;margin:10px 0;'>"
        f"⬇ {len(intel_kv)} LLM-extracted fields · modified values included</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📥 Download JSON",
            data=full_json_str,
            file_name="extracted_fields.json",
            mime="application/json",
            use_container_width=True,
        )
    with c2:
        if st.button("📋 Copy to clipboard", use_container_width=True):
            st.toast("Copied!")
            st.session_state["_json_clipboard"] = full_json_str

    # ── Full raw text (collapsed) ─────────────────────────────────────────────
    full_text = intelligence.get("full_text", "")
    if full_text:
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        with st.expander("📄 Full extracted text"):
            st.text_area("raw_text_area", value=full_text, height=300,
                         label_visibility="collapsed")
            st.download_button(
                "📥 Download raw text", data=full_text,
                file_name="extracted_text.txt", mime="text/plain",
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — TRANSFORMATION JOURNEY
# ─────────────────────────────────────────────────────────────────────────────

def _render_journey_tab(
    intelligence: dict,
    selected_sheet: str,
    uploaded_name: str,
) -> None:
    intel_fields   = _get_intelligence_entities(selected_sheet)
    raw_fields     = _get_raw_fields(selected_sheet)
    display_fields = intel_fields if intel_fields else raw_fields

    eds           = _edits()
    hist          = _edit_history()
    session_start = st.session_state.get("_session_start", "")
    now_str       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    changed_fields   = [(fn, fi) for fn, fi in display_fields if fn in hist and hist[fn]]
    unchanged_fields = [(fn, fi) for fn, fi in display_fields if fn not in hist or not hist[fn]]
    edit_count       = len(changed_fields)

    last_edit_ts = ""
    if edit_count:
        all_ts = [ch["timestamp"] for fn, _ in changed_fields
                  for ch in hist.get(fn, []) if ch.get("timestamp")]
        if all_ts:
            last_edit_ts = max(all_ts)[:19].replace("T", " ")

    # Pipeline trace header
    st.markdown(
        f"<div style='background:#12121c;border:1px solid #2a2a45;"
        f"border-radius:10px;padding:16px 20px;margin-bottom:20px;'>"
        f"<div style='font-size:10px;font-weight:700;color:#f5c842;"
        f"font-family:monospace;text-transform:uppercase;letter-spacing:2px;"
        f"margin-bottom:14px;'>⚡ Pipeline Trace</div>"
        f"<div style='display:grid;grid-template-columns:160px 1fr;gap:8px;"
        f"padding:8px 0;border-bottom:1px solid #1a1a2e;align-items:start;'>"
        f"<span style='font-size:10px;font-weight:700;color:#f5c842;"
        f"font-family:monospace;text-transform:uppercase;letter-spacing:.8px;'>"
        f"📄 FILE PARSED</span>"
        f"<span style='font-size:11px;color:#c8c7f0;font-family:monospace;'>"
        f"→ Fields extracted from uploaded PDF &nbsp;"
        f"<span style='color:#555;'>{session_start[:19].replace('T',' ') if session_start else now_str}</span>"
        f"</span></div>"
        f"<div style='display:grid;grid-template-columns:160px 1fr;gap:8px;"
        f"padding:8px 0;align-items:start;'>"
        f"<span style='font-size:10px;font-weight:700;color:#4f9cf9;"
        f"font-family:monospace;text-transform:uppercase;letter-spacing:.8px;'>"
        f"✏️ USER EDITS</span>"
        f"<span style='font-size:11px;color:#c8c7f0;font-family:monospace;'>"
        + (
            f"→ {edit_count} field(s) manually updated &nbsp;"
            f"<span style='color:#555;'>{last_edit_ts}</span>"
            if edit_count else
            f"→ <span style='color:#555;'>No edits made this session</span>"
        )
        + f"</span></div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(_section_header("Field Transformation Timeline"), unsafe_allow_html=True)

    def _step_circle(n: int, color: str) -> str:
        return (
            f"<div style='width:26px;height:26px;border-radius:50%;"
            f"background:{color}22;border:2px solid {color};"
            f"display:flex;align-items:center;justify-content:center;"
            f"font-size:11px;font-weight:700;color:{color};"
            f"font-family:monospace;flex-shrink:0;'>{n}</div>"
        )

    def _field_card(fname: str, finfo: dict) -> None:
        extracted = finfo.get("value", "")
        changes   = hist.get(fname, [])
        is_mod    = bool(changes)
        border    = "#f5c842" if is_mod else "#2a2a45"
        bg        = "#1a1500" if is_mod else "#12121c"
        mod_badge = (
            "<span style='margin-left:8px;font-size:9px;font-weight:700;"
            "color:#f5c842;background:#f5c84215;border:1px solid #f5c84250;"
            "border-radius:10px;padding:2px 8px;font-family:monospace;'>"
            "MODIFIED</span>"
            if is_mod else ""
        )
        src_page = finfo.get("source_page", "")
        src_text = finfo.get("source_text", "")
        step1_ts = session_start[:19].replace("T", " ") if session_start else now_str

        html = (
            f"<div style='background:{bg};border:1px solid {border};"
            f"border-radius:10px;padding:16px 18px;margin-bottom:12px;'>"
            f"<div style='font-size:12px;font-weight:700;color:#f0efff;"
            f"font-family:monospace;text-transform:uppercase;letter-spacing:1px;"
            f"margin-bottom:14px;'>{fname}{mod_badge}</div>"
            # Step 1 — Extracted
            f"<div style='display:flex;gap:12px;margin-bottom:10px;'>"
            f"{_step_circle(1,'#34d399')}"
            f"<div style='flex:1;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"margin-bottom:4px;'>"
            f"<span style='font-size:10px;font-weight:700;color:#34d399;"
            f"font-family:monospace;text-transform:uppercase;'>Extracted from Document</span>"
            f"<span style='font-size:9px;color:#555;font-family:monospace;'>"
            f"⏱ {step1_ts} · pdf_azure_parser</span>"
            f"</div>"
            f"<div style='font-size:11px;color:#8888bb;font-family:monospace;"
            f"margin-bottom:5px;'>{'Page ' + str(src_page) if src_page else 'PDF extraction'}</div>"
            f"<div style='background:#0d0d14;border:1px solid #1e1e30;border-radius:5px;"
            f"padding:8px 12px;font-size:12px;color:#f0efff;font-family:monospace;"
            f"word-break:break-word;min-height:32px;'>"
            f"{extracted if extracted else '<span style=\"color:#3a3a55;\">—</span>'}"
            f"</div>"
            + (
                f"<div style='font-size:10px;color:#555;font-family:monospace;"
                f"background:#0a0a12;border-left:2px solid #2a2a45;padding:4px 8px;"
                f"margin-top:5px;border-radius:0 4px 4px 0;font-style:italic;'>"
                f"📄 {src_text}</div>"
                if src_text else ""
            )
            + f"</div></div>"
            # Step 2 — LLM / Direct
            f"<div style='display:flex;gap:12px;margin-bottom:10px;'>"
            f"{_step_circle(2,'#4f9cf9')}"
            f"<div style='flex:1;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"margin-bottom:4px;'>"
            f"<span style='font-size:10px;font-weight:700;color:#4f9cf9;"
            f"font-family:monospace;text-transform:uppercase;'>→ Direct (LLM)</span>"
            f"<span style='font-size:9px;color:#555;font-family:monospace;'>"
            f"⏱ {step1_ts} · pdf_intelligence</span>"
            f"</div>"
            f"<div style='font-size:11px;color:#8888bb;font-family:monospace;'>"
            f"Extracted by AI — type-specific field list applied</div>"
            f"</div></div>"
        )

        for i, ch in enumerate(changes):
            ts     = (ch.get("timestamp", "")[:19] or "").replace("T", " ")
            from_v = ch.get("from", "")
            to_v   = ch.get("to", "")
            html += (
                f"<div style='display:flex;gap:12px;margin-bottom:8px;'>"
                f"{_step_circle(i+3,'#f5c842')}"
                f"<div style='flex:1;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"margin-bottom:6px;'>"
                f"<span style='font-size:10px;font-weight:700;color:#f5c842;"
                f"font-family:monospace;text-transform:uppercase;'>→ User Edit</span>"
                f"<span style='font-size:9px;color:#555;font-family:monospace;'>"
                f"⏱ {ts} · _sync_edit()</span>"
                f"</div>"
                f"<div style='display:flex;gap:10px;align-items:center;'>"
                f"<div style='flex:1;background:#1a0000;border:1px solid #f8717140;"
                f"border-radius:5px;padding:7px 12px;font-size:12px;"
                f"color:#f87171;font-family:monospace;word-break:break-word;'>"
                f"FROM: {from_v or '—'}</div>"
                f"<span style='font-size:16px;color:#555;'>→</span>"
                f"<div style='flex:1;background:#001a0a;border:1px solid #34d39940;"
                f"border-radius:5px;padding:7px 12px;font-size:12px;"
                f"color:#34d399;font-family:monospace;word-break:break-word;'>"
                f"TO: {to_v or '—'}</div>"
                f"</div></div></div>"
            )

        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    for fname, finfo in changed_fields:
        _field_card(fname, finfo)

    if unchanged_fields:
        with st.expander(f"📋 {len(unchanged_fields)} unchanged field(s)"):
            for fname, finfo in unchanged_fields:
                _field_card(fname, finfo)

    # Audit Log
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown(_section_header("Audit Log"), unsafe_allow_html=True)

    _EVENT_META = {
        "FILE_INGESTED":           {"color": "#34d399", "icon": "📄"},
        "SHEET_PARSED":            {"color": "#4f9cf9", "icon": "🔍"},
        "SHEET_LOADED_FROM_CACHE": {"color": "#a78bfa", "icon": "💾"},
    }

    try:
        from modules.audit import _load_audit_log  # type: ignore[import]
        full_log = _load_audit_log()

        def _is_cur(e: dict) -> bool:
            ts = e.get("timestamp", "")
            return not ts or not session_start or ts >= session_start

        def _is_rel(e: dict) -> bool:
            return (
                uploaded_name in (e.get("filename") or "")
                or "PDF" in (e.get("event") or "").upper()
                or "pdf" in (e.get("sheet_type") or "").lower()
            )

        cur_log  = [e for e in full_log if _is_cur(e) and _is_rel(e)]
        hist_log = [e for e in full_log if not _is_cur(e) and _is_rel(e)]

        def _log_row(entry: dict, idx: int, prefix: str) -> None:
            ts    = (entry.get("timestamp", "")[:19] or "").replace("T", " ")
            event = entry.get("event", "UNKNOWN")
            em    = _EVENT_META.get(event, {"color": "#6b7280", "icon": "●"})
            c     = em["color"]
            parts = []
            if entry.get("sheet"):
                parts.append(entry["sheet"])
            if entry.get("sheet_type"):
                parts.append(entry["sheet_type"])
            if entry.get("claim_rows"):
                parts.append(f"{entry['claim_rows']} rows")
            detail = " · ".join(parts)
            st.markdown(
                f"<div style='background:#12121c;border:1px solid #1e1e30;"
                f"border-left:3px solid {c};border-radius:6px;"
                f"padding:9px 14px;margin-bottom:4px;"
                f"display:flex;align-items:center;gap:12px;'>"
                f"<span style='font-size:9px;font-weight:700;color:{c};"
                f"font-family:monospace;background:{c}14;border:1px solid {c}40;"
                f"border-radius:4px;padding:2px 8px;white-space:nowrap;"
                f"text-transform:uppercase;'>{em['icon']} {event}</span>"
                f"<span style='font-size:10px;color:#555;font-family:monospace;"
                f"white-space:nowrap;'>· {ts}</span>"
                f"<span style='font-size:11px;color:#8888bb;font-family:monospace;'>"
                f"· {detail}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if not cur_log:
            st.info("No log entries for this session.")
        else:
            for i, e in enumerate(reversed(cur_log[-30:])):
                _log_row(e, i, "cur")

        if hist_log:
            st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
            _hk = "_audit_show_hist"
            show_h = st.session_state.get(_hk, False)
            if st.button(
                "🕑 Hide previous history" if show_h else "🕑 View history",
                key="toggle_audit_hist",
            ):
                st.session_state[_hk] = not show_h
                st.rerun()
            if show_h:
                st.markdown(
                    f"<div style='font-size:11px;color:{_LBL};font-family:monospace;margin:8px 0;'>"
                    f"{len(hist_log)} previous session entries</div>",
                    unsafe_allow_html=True,
                )
                for i, e in enumerate(reversed(hist_log[-30:])):
                    _log_row(e, i, "hist")

    except Exception as exc:
        st.warning(f"Could not load audit log: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — AI ASSESSMENT  (4 judge cards only)
# ─────────────────────────────────────────────────────────────────────────────

def _assessment_card(icon: str, title: str, content: str, color: str, badge: str = "") -> str:
    return (
        f"<div style='background:#0d0d1a;border:1px solid {color}35;"
        f"border-left:4px solid {color};border-radius:10px;"
        f"padding:18px 20px;margin-bottom:14px;'>"
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:12px;'>"
        f"<span style='font-size:22px;'>{icon}</span>"
        f"<span style='font-size:12px;font-weight:700;color:{color};"
        f"font-family:monospace;text-transform:uppercase;letter-spacing:1.2px;'>{title}</span>"
        + (
            f"<span style='margin-left:auto;font-size:10px;color:{color};"
            f"background:{color}18;border:1px solid {color}40;border-radius:20px;"
            f"padding:2px 10px;font-family:monospace;white-space:nowrap;'>{badge}</span>"
            if badge else ""
        )
        + f"</div>"
        f"<div style='font-size:13px;color:{_TXT};line-height:1.85;'>{content}</div>"
        f"</div>"
    )


def _render_ai_assessment_tab(intelligence: dict) -> None:
    judge   = intelligence.get("analysis", {}).get("judge", {})
    signals = intelligence.get("analysis", {}).get("signals", [])

    st.markdown(_section_header("AI Verdict"), unsafe_allow_html=True)

    cards = [
        ("🧠", "Classification Rationale",  judge.get("classification_reasoning", ""),
         "#4f9cf9", "AI Verified"),
        ("⚡", "Signal Validation",          judge.get("signal_validation",        ""),
         "#f5c842", f"{len(signals)} signal(s) reviewed"),
        ("📊", "Data Quality Assessment",    judge.get("data_quality",             ""),
         "#34d399", "Quality Review"),
        ("🎯", "Recommendations",            judge.get("recommendations",          ""),
         "#a78bfa", "Action Required"),
    ]

    any_content = False
    for icon, title, content, color, badge in cards:
        if not content:
            continue
        any_content = True
        st.markdown(_assessment_card(icon, title, content, color, badge=badge),
                    unsafe_allow_html=True)

    if not any_content:
        st.info("AI assessment not available — LLM may not have completed analysis.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def render_pdf_analysis_panel(
    intelligence: dict,
    uploaded_name: str,
    selected_sheet: str,
) -> None:
    st.markdown(_UPLOADER_PLUS_CSS, unsafe_allow_html=True)

    doc_type = intelligence.get("doc_type", "Legal")
    meta     = _DOC_TYPE_META.get(doc_type, _DOC_TYPE_META["Legal"])

    # Invalidate ADI lookup on new file
    if st.session_state.get("_adi_lookup_file") != uploaded_name:
        st.session_state.pop("_adi_lookup", None)
        st.session_state["_adi_lookup_file"] = uploaded_name

    # Resolve PDF path
    _tmpdir  = st.session_state.get("tmpdir", "")
    pdf_path: str | None = None
    if _tmpdir:
        for _ext in (".pdf", ".PDF"):
            _c = os.path.join(_tmpdir, f"input{_ext}")
            if os.path.exists(_c):
                pdf_path = _c
                break

    st.markdown(
        f"<div style='background:{meta['bg']};border:1px solid {meta['color']}30;"
        f"border-radius:10px;padding:13px 18px;margin-bottom:14px;'>"
        f"<div style='display:flex;align-items:center;gap:12px;'>"
        f"<span style='font-size:22px;'>{meta['icon']}</span>"
        f"<div>"
        f"<div style='font-size:14px;font-weight:700;color:{meta['color']};"
        f"font-family:monospace;text-transform:uppercase;letter-spacing:1.5px;'>"
        f"{doc_type} Document Analysis</div>"
        f"<div style='font-size:11px;color:{_LBL};margin-top:3px;'>"
        f"📄 {uploaded_name} · {selected_sheet} · "
        f"{intelligence.get('page_count', 0)} page(s)</div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    intel_fields    = _get_intelligence_entities(selected_sheet)
    _entities_count = len(intel_fields) if intel_fields else len(_get_raw_fields(selected_sheet))
    _signals_count  = len(intelligence.get("analysis", {}).get("signals", []))

    tabs = st.tabs([
        f"🔍 Entities ({_entities_count})",
        "📝 Summary",
        f"⚡ Signals ({_signals_count})",
        "📄 Raw JSON",
        "🔄 Transformation Journey",
        "🧑‍⚖️ AI Assessment",
    ])

    with tabs[0]:
        _render_entities_tab(intelligence, selected_sheet, pdf_path)
    with tabs[1]:
        _render_summary_tab(intelligence, selected_sheet)
    with tabs[2]:
        _render_signals_tab(intelligence)
    with tabs[3]:
        _render_raw_json_tab(intelligence, selected_sheet)
    with tabs[4]:
        _render_journey_tab(intelligence, selected_sheet, uploaded_name)
    with tabs[5]:
        _render_ai_assessment_tab(intelligence)
