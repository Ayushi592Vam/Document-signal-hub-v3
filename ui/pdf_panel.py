"""
ui/pdf_panel.py
Sidebar PDF upload + results panel.
v3: KV-only export, no paragraphs tab, spatial fallback notice.
"""

import json
import datetime

import streamlit as st

from modules.pdf_extractor import extract_pdf, flatten_for_export, render_page_with_boxes


PDF_CSS = """
<style>
.pdf-section-lbl {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.2px; color: var(--t3, #888);
    font-family: var(--mono, monospace); margin: 14px 0 6px 0;
}
.pdf-meta-bar  { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
.pdf-meta-chip {
    background: var(--s0, #1a1a2e); border: 1px solid var(--b0, #2a2a3e);
    border-radius: 20px; padding: 3px 10px; font-size: 11px;
    color: var(--t2, #aaa); font-family: var(--mono, monospace);
}
.pdf-meta-chip span { color: var(--blue, #4f9cf9); font-weight: 700; }
.pdf-meta-chip.spatial span { color: var(--yellow, #f0c040); }

.pdf-kv-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 16px; }
.pdf-kv-table th {
    background: var(--s0, #1a1a2e); color: var(--t3, #888);
    font-family: var(--mono, monospace); font-size: 10px;
    text-transform: uppercase; letter-spacing: 1px;
    padding: 6px 10px; text-align: left;
    border-bottom: 1px solid var(--b0, #2a2a3e);
}
.pdf-kv-table td {
    padding: 6px 10px; border-bottom: 1px solid var(--b0, #1e1e2e);
    color: var(--t0, #eee); vertical-align: top; word-break: break-word;
}
.pdf-kv-table tr:hover td { background: rgba(79,156,249,0.06); }
.pdf-kv-key    { color: var(--blue,   #4f9cf9) !important; font-family: var(--mono, monospace); font-size: 11px !important; }
.pdf-kv-val    { color: var(--t0,     #eee)    !important; }
.pdf-kv-conf   { color: var(--t3,     #888)    !important; font-family: var(--mono, monospace); font-size: 10px !important; text-align: right !important; }
.pdf-kv-pg     { color: var(--t3,     #666)    !important; font-family: var(--mono, monospace); font-size: 10px !important; text-align: center !important; }
.pdf-kv-src    { font-size: 9px !important; color: var(--yellow, #f0c040) !important; font-family: var(--mono, monospace) !important; }

.pdf-tbl-wrap  { overflow-x: auto; margin-bottom: 16px; border: 1px solid var(--b0, #2a2a3e); border-radius: 7px; }
.pdf-tbl-grid  { width: 100%; border-collapse: collapse; font-size: 11px; font-family: var(--mono, monospace); }
.pdf-tbl-grid th { background: var(--s0, #1a1a2e); color: var(--blue, #4f9cf9); padding: 5px 9px; border-bottom: 1px solid var(--b0, #2a2a3e); text-align: left; font-weight: 600; }
.pdf-tbl-grid td { padding: 5px 9px; border-bottom: 1px solid var(--b0, #1e1e2e); color: var(--t1, #ccc); }

.pdf-json-panel  { background: var(--s0, #0f0f1a); border: 1px solid var(--b0, #2a2a3e); border-radius: 8px; overflow: hidden; margin-bottom: 14px; }
.pdf-json-header { display: flex; justify-content: space-between; align-items: center; padding: 7px 12px; background: var(--s1, #16162a); border-bottom: 1px solid var(--b0, #2a2a3e); }
.pdf-json-body   { padding: 12px; font-size: 11px; font-family: var(--mono, monospace); color: var(--green, #34d399); white-space: pre-wrap; overflow-y: auto; max-height: 420px; }

.pdf-error-banner {
    background: rgba(248,113,113,0.09); border: 1px solid rgba(248,113,113,0.3);
    border-left: 3px solid var(--red, #f87171); border-radius: 7px;
    padding: 12px 14px; margin-bottom: 12px; font-size: 12px;
    color: var(--red, #f87171); font-family: var(--mono, monospace);
}
.pdf-spatial-banner {
    background: rgba(240,192,64,0.08); border: 1px solid rgba(240,192,64,0.25);
    border-left: 3px solid var(--yellow, #f0c040); border-radius: 7px;
    padding: 8px 12px; margin-bottom: 10px; font-size: 11px;
    color: var(--yellow, #f0c040); font-family: var(--mono, monospace);
}
.bbox-legend { display: flex; gap: 14px; font-size: 11px; font-family: var(--mono, monospace); margin-bottom: 8px; flex-wrap: wrap; }
.bbox-dot    { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
def render_pdf_sidebar() -> None:
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div style='font-size:11px;font-weight:700;color:var(--t2,#aaa);"
        "font-family:var(--mono,monospace);text-transform:uppercase;"
        "letter-spacing:1px;margin-bottom:8px;'>📄 PDF / Image Extractor</div>",
        unsafe_allow_html=True,
    )
    pdf_file = st.file_uploader(
        "Upload PDF or image",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
        key="pdf_uploader",
        label_visibility="collapsed",
    )
    if not pdf_file:
        st.sidebar.markdown(
            "<div style='font-size:10px;color:var(--t3,#666);font-family:monospace;'>"
            "Supports PDFs, scanned images, FNOLs, legal docs</div>",
            unsafe_allow_html=True,
        )
        return

    _fp = f"{pdf_file.name}_{pdf_file.file_id}"
    if st.session_state.get("_pdf_last_fp") == _fp:
        res = st.session_state.get("_pdf_result", {})
        if res and not res.get("error"):
            kv_n = len(res.get("kv_pairs", []))
            spatial_n = sum(1 for kv in res.get("kv_pairs", []) if kv.get("_source", "adi") != "adi")
            note = f" ({spatial_n} spatial)" if spatial_n else ""
            st.sidebar.markdown(
                f"<div style='font-size:10px;color:var(--green,#34d399);font-family:monospace;'>"
                f"✓ {res['filename']}<br>{kv_n} KV{note}</div>",
                unsafe_allow_html=True,
            )
        elif res and res.get("error"):
            st.sidebar.markdown(
                f"<div style='font-size:10px;color:var(--red,#f87171);font-family:monospace;'>"
                f"⚠ {res['error'][:80]}</div>",
                unsafe_allow_html=True,
            )
        return

    with st.sidebar:
        with st.spinner("Extracting with Azure ADI…"):
            file_bytes = pdf_file.read()
            result     = extract_pdf(file_bytes, pdf_file.name)

    st.session_state["_pdf_result"]  = result
    st.session_state["_pdf_last_fp"] = _fp
    for k in ["pdf_show_json", "_pdf_bbox_page", "_pdf_bbox_hi"]:
        st.session_state.pop(k, None)
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
def render_pdf_results() -> None:
    result = st.session_state.get("_pdf_result")
    if not result:
        return

    st.markdown(PDF_CSS, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p class='pdf-section-lbl'>📄 PDF Extraction Results</p>", unsafe_allow_html=True)

    if result.get("error"):
        st.markdown(f"<div class='pdf-error-banner'>⚠ {result['error']}</div>", unsafe_allow_html=True)
        if "credentials not found" in result["error"]:
            st.info("Add **AZURE_DI_ENDPOINT** and **AZURE_DI_KEY** to your `.env` file then re-upload.", icon="🔑")
        return

    kv_pairs  = result.get("kv_pairs",   [])
    tbl_n     = len(result.get("tables", []))
    pg_n      = result.get("page_count", 0)
    kv_n      = len(kv_pairs)
    spatial_n = sum(1 for kv in kv_pairs if kv.get("_source", "adi") != "adi")
    adi_n     = kv_n - spatial_n

    st.markdown(
        f"<div class='pdf-meta-bar'>"
        f"<div class='pdf-meta-chip'>📄 <span>{result['filename']}</span></div>"
        f"<div class='pdf-meta-chip'>Pages <span>{pg_n}</span></div>"
        f"<div class='pdf-meta-chip'>KV pairs <span>{kv_n}</span></div>"
        + (f"<div class='pdf-meta-chip spatial'>ADI <span>{adi_n}</span> · Spatial <span>{spatial_n}</span></div>" if spatial_n else "")
        + f"<div class='pdf-meta-chip'>Tables <span>{tbl_n}</span></div>"
        f"<div class='pdf-meta-chip'>Extracted <span>{result.get('extracted_at','')[:19].replace('T',' ')}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Spatial fallback notice
    if spatial_n:
        st.markdown(
            f"<div class='pdf-spatial-banner'>"
            f"⚡ {spatial_n} field{'s' if spatial_n > 1 else ''} recovered via spatial layout analysis "
            f"(ADI did not detect them as form fields). Verify values before use."
            f"</div>",
            unsafe_allow_html=True,
        )

    # Build tabs — no paragraphs tab
    tab_labels = ["🔑 Key-Value Pairs", "🗺 Bounding Boxes"]
    if tbl_n:
        tab_labels.append("📊 Tables")
    tab_labels.append("{ } JSON Export")

    tabs    = st.tabs(tab_labels)
    tab_idx = 0

    # ── KV Pairs ──────────────────────────────────────────────────────────────
    with tabs[tab_idx]:
        tab_idx += 1
        if not kv_pairs:
            st.markdown(
                "<div style='color:var(--t3,#888);font-size:12px;font-family:monospace;padding:16px 0;'>"
                "No key-value pairs detected.</div>",
                unsafe_allow_html=True,
            )
        else:
            _kv_search = st.text_input(
                "Filter…", key="pdf_kv_search",
                placeholder="Type to filter keys or values…",
                label_visibility="collapsed",
            )
            filtered = kv_pairs
            if _kv_search:
                q = _kv_search.lower()
                filtered = [p for p in kv_pairs if q in p["key"].lower() or q in p["value"].lower()]

            rows_html = "".join(
                f"<tr>"
                f"<td class='pdf-kv-pg'>p{p.get('page',1)}</td>"
                f"<td class='pdf-kv-key'>{_esc(p['key'])}"
                + (f"<br><span class='pdf-kv-src'>spatial</span>" if p.get("_source","adi") != "adi" else "")
                + f"</td>"
                f"<td class='pdf-kv-val'>{_esc(p['value']) or '<span style=\"color:var(--t3)\">—</span>'}</td>"
                f"<td class='pdf-kv-conf'>{p['confidence']}%</td>"
                f"</tr>"
                for p in filtered
            )
            st.markdown(
                f"<table class='pdf-kv-table'><thead><tr>"
                f"<th>Pg</th><th>Key</th><th>Value</th><th style='text-align:right;'>Conf.</th>"
                f"</tr></thead><tbody>{rows_html}</tbody></table>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:10px;color:var(--t3,#777);font-family:monospace;margin-top:-8px;'>"
                f"Showing {len(filtered)} of {kv_n} pairs</div>",
                unsafe_allow_html=True,
            )

    # ── Bounding Boxes ────────────────────────────────────────────────────────
    with tabs[tab_idx]:
        tab_idx += 1
        _deps_ok = _check_render_deps()

        if not _deps_ok:
            st.info(
                "Bounding box rendering requires **PyMuPDF** and **Pillow**.\n\n"
                "```\npip install pymupdf Pillow\n```\n\nRestart Streamlit after installing.",
                icon="📦",
            )
        elif not result.get("file_bytes"):
            st.warning("Original file bytes not available for rendering.")
        else:
            page_nums = sorted({kv.get("page", 1) for kv in kv_pairs}) or list(range(1, pg_n + 1))

            col_pg, col_hi, col_dpi = st.columns([1, 2, 1])
            with col_pg:
                sel_page = st.selectbox("Page", page_nums, key="pdf_bbox_page_sel")
            kv_on_page = [(i, kv) for i, kv in enumerate(kv_pairs) if kv.get("page") == sel_page]
            with col_hi:
                hi_opts = ["— Show all —"] + [
                    f"{kv['key'][:40]} → {kv['value'][:30]}" for _, kv in kv_on_page
                ]
                hi_sel = st.selectbox("Highlight field", hi_opts, key="pdf_bbox_hi_sel")
            with col_dpi:
                dpi = st.selectbox("Resolution", [100, 150, 200], index=1, key="pdf_bbox_dpi")

            highlight_idx = None
            if hi_sel != "— Show all —":
                chosen_local = hi_opts.index(hi_sel) - 1
                if 0 <= chosen_local < len(kv_on_page):
                    highlight_idx = kv_on_page[chosen_local][0]

            st.markdown(
                "<div class='bbox-legend'>"
                "<span><span class='bbox-dot' style='background:rgba(79,156,249,0.7);'></span>"
                "<span style='color:var(--t2,#aaa);'>Key region</span></span>"
                "<span><span class='bbox-dot' style='background:rgba(52,211,153,0.7);'></span>"
                "<span style='color:var(--t2,#aaa);'>Value region</span></span>"
                "<span><span class='bbox-dot' style='background:rgba(255,220,50,0.8);'></span>"
                "<span style='color:var(--t2,#aaa);'>Highlighted key</span></span>"
                "<span><span class='bbox-dot' style='background:rgba(255,140,0,0.8);'></span>"
                "<span style='color:var(--t2,#aaa);'>Highlighted value</span></span>"
                "</div>",
                unsafe_allow_html=True,
            )

            with st.spinner("Rendering page…"):
                png_bytes = render_page_with_boxes(result, sel_page, highlight_idx, dpi)

            if png_bytes:
                st.image(png_bytes, use_container_width=True,
                         caption=f"Page {sel_page} — {len(kv_on_page)} KV regions")
                st.download_button(
                    "📥 Download annotated page", data=png_bytes,
                    file_name=f"{result['filename']}_p{sel_page}_bbox.png",
                    mime="image/png", key="pdf_bbox_dl",
                )
            else:
                st.warning("Could not render page image.")

            if kv_on_page:
                st.markdown(
                    f"<div style='font-size:10px;color:var(--t3,#888);font-family:monospace;"
                    f"margin-top:12px;margin-bottom:4px;'>{len(kv_on_page)} fields on page {sel_page}</div>",
                    unsafe_allow_html=True,
                )
                rows_html = "".join(
                    f"<tr>"
                    f"<td class='pdf-kv-key'>{_esc(kv['key'])}"
                    + (f"<br><span class='pdf-kv-src'>spatial</span>" if kv.get("_source","adi") != "adi" else "")
                    + f"</td>"
                    f"<td class='pdf-kv-val'>{_esc(kv['value']) or '<span style=\"color:var(--t3)\">—</span>'}</td>"
                    f"<td class='pdf-kv-conf'>{kv['confidence']}%</td>"
                    f"</tr>"
                    for _, kv in kv_on_page
                )
                st.markdown(
                    f"<table class='pdf-kv-table'><thead><tr>"
                    f"<th>Key</th><th>Value</th><th style='text-align:right;'>Conf.</th>"
                    f"</tr></thead><tbody>{rows_html}</tbody></table>",
                    unsafe_allow_html=True,
                )

    # ── Tables (if any) ───────────────────────────────────────────────────────
    if tbl_n:
        with tabs[tab_idx]:
            tab_idx += 1
            for t_idx, tbl in enumerate(result.get("tables", [])):
                rows = tbl["rows"]
                if not rows:
                    continue
                st.markdown(
                    f"<div style='font-size:10px;color:var(--t3,#888);font-family:monospace;margin-bottom:6px;'>"
                    f"Table {t_idx+1} · Page {tbl['page']} · {len(rows)} rows × {len(rows[0])} cols</div>",
                    unsafe_allow_html=True,
                )
                header  = rows[0]
                body    = rows[1:]
                th_html = "".join(f"<th>{_esc(h)}</th>" for h in header)
                tr_html = "".join(
                    "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>"
                    for row in body
                )
                st.markdown(
                    f"<div class='pdf-tbl-wrap'><table class='pdf-tbl-grid'>"
                    f"<thead><tr>{th_html}</tr></thead><tbody>{tr_html}</tbody>"
                    f"</table></div>",
                    unsafe_allow_html=True,
                )

    # ── JSON Export (KV only) ─────────────────────────────────────────────────
    with tabs[tab_idx]:
        export_payload = flatten_for_export(result)
        json_str       = json.dumps(export_payload, indent=2, ensure_ascii=False)

        if "pdf_show_json" not in st.session_state:
            st.session_state["pdf_show_json"] = True
        _lbl = "▲ Hide Preview" if st.session_state["pdf_show_json"] else "{ } Show Preview"
        if st.button(_lbl, key="pdf_json_toggle"):
            st.session_state["pdf_show_json"] = not st.session_state["pdf_show_json"]

        if st.session_state["pdf_show_json"]:
            st.markdown(
                f"<div class='pdf-json-panel'>"
                f"<div class='pdf-json-header'>"
                f"<span style='font-size:10px;font-weight:600;color:var(--t2,#aaa);font-family:monospace;'>"
                f"⬡ {result['filename']}</span>"
                f"<span style='font-size:10px;color:var(--t3,#666);font-family:monospace;'>"
                f"{kv_n} key-value pairs</span></div>"
                f"<div class='pdf-json-body'>{_esc(json_str)}</div></div>",
                unsafe_allow_html=True,
            )

        safe_name = result["filename"].rsplit(".", 1)[0].replace(" ", "_")
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"{safe_name}_kv_{datetime.date.today()}.json",
            mime="application/json",
            use_container_width=True,
            key="pdf_download_btn",
            type="primary",
        )

    # ── Clear ─────────────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("✕ Clear PDF Result", key="pdf_clear_btn"):
        for k in ["_pdf_result", "_pdf_last_fp", "pdf_show_json"]:
            st.session_state.pop(k, None)
        st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _esc(text: str) -> str:
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _check_render_deps() -> bool:
    try:
        import fitz
        from PIL import Image
        return True
    except ImportError:
        return False