import json
import io
import pandas as pd
import streamlit as st
from pypdf import PdfReader

from model_openai import extract_information_for_documents
from credit_cost_calc import get_credits_cost

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="AI Document Info Extractor",
    page_icon="⚙️",
    layout="wide",
)

# -------------------------------------------------
# Authentication Gate
# -------------------------------------------------
def check_password() -> bool:
    """Gate the app behind the shared secret in st.secrets['USER_PWD']."""
    if st.session_state.get("authenticated"):
        return True

    st.title("AI Accountant - Smart Diff Checker")
    secret = st.text_input("Secret", type="password", key="secret_input")
    if st.button("Enter"):
        if secret == st.secrets.get("USER_PWD"):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Wrong secret.")
    return False

# Stop execution if user is not authenticated
if not check_password():
    st.stop()

# -------------------------------------------------
# Session state
# -------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

_DEFAULT_FIELDS = [{"field": "", "type": "String", "description": ""}]
_DEFAULT_INSTRUCTIONS = [{"instruction": ""}]

if "fields_df" not in st.session_state:
    st.session_state.fields_df = pd.DataFrame(_DEFAULT_FIELDS)

if "instructions_df" not in st.session_state:
    st.session_state.instructions_df = pd.DataFrame(_DEFAULT_INSTRUCTIONS)

# Apply delta edits for fields_df
if "fields_editor" in st.session_state:
    _es = st.session_state["fields_editor"]
    try:
        _df = st.session_state.fields_df.copy()

        for _ri, _changes in _es.get("edited_rows", {}).items():
            _r = int(_ri)
            for _col, _val in _changes.items():
                if _r < len(_df):
                    _df.at[_r, _col] = _val

        for _new in _es.get("added_rows", []):
            _entry = {"field": "", "type": "String", "description": ""}
            _entry.update(_new)
            _df = pd.concat([_df, pd.DataFrame([_entry])], ignore_index=True)

        for _di in sorted(_es.get("deleted_rows", []), reverse=True):
            if _di < len(_df):
                _df = _df.drop(index=_di).reset_index(drop=True)

        st.session_state.fields_df = _df
    except Exception:
        pass

# Apply delta edits for instructions_df
if "instructions_editor" in st.session_state:
    _es = st.session_state["instructions_editor"]
    try:
        _df = st.session_state.instructions_df.copy()

        for _ri, _changes in _es.get("edited_rows", {}).items():
            _r = int(_ri)
            for _col, _val in _changes.items():
                if _r < len(_df):
                    _df.at[_r, _col] = _val

        for _new in _es.get("added_rows", []):
            _entry = {"instruction": ""}
            _entry.update(_new)
            _df = pd.concat([_df, pd.DataFrame([_entry])], ignore_index=True)

        for _di in sorted(_es.get("deleted_rows", []), reverse=True):
            if _di < len(_df):
                _df = _df.drop(index=_di).reset_index(drop=True)

        st.session_state.instructions_df = _df
    except Exception:
        pass

if "extraction_output" not in st.session_state:
    st.session_state.extraction_output = None

if "trigger_extraction" not in st.session_state:
    st.session_state.trigger_extraction = False

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Luna"


# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown("""
<style>

/* HEADER TEXT */
.app-title {
    font-size: 2.2rem;
    font-weight: 700;
}

/* SECTION TITLES */
.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

/* SIDEBAR STEPS */
.sidebar-step {
    padding: 0.6rem 0;
    font-weight: 600;
}

.sidebar-step.active {
    color: #0A66C2;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ AI Document Info Extractor")
    st.caption("Internal Tool (OpenAI Powered)")

    steps = ["1. Setup", "2. Summary", "3. Results"]

    for i, label in enumerate(steps, start=1):
        css = "sidebar-step active" if st.session_state.step == i else "sidebar-step"
        st.markdown(f'<div class="{css}">{label}</div>', unsafe_allow_html=True)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown('<div class="app-title">⚙️ AI Document Info Extractor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload documents and define the data you want to extract</div>',
    unsafe_allow_html=True,
)

# =================================================
# STEP 1 — SETUP
# =================================================
if st.session_state.step == 1:

    # ---- Row 1: Upload (left) + Advanced Options (right) ----
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col_upload, col_advanced = st.columns([1.2, 1])

    with col_upload:
        st.markdown('<div class="section-title">📤 Upload documents</div>', unsafe_allow_html=True)
        st.caption("Upload your PDF documents for independent AI field extraction.")

        files = st.file_uploader(
            "PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if files:
            st.session_state.uploaded_files = files

        if st.session_state.uploaded_files:
            st.info(f"{len(st.session_state.uploaded_files)} document(s) selected")
            for f in st.session_state.uploaded_files:
                st.caption(f"📄 {f.name}")

    with col_advanced:
        st.markdown('<div class="section-title">⚙️ AI Model Selection</div>', unsafe_allow_html=True)
        st.caption('Toggle between OpenAI extraction models:')
        st.caption('- **Luna** (Default): Fast & highly cost-effective (Powered by `gpt-4o-mini`).')
        st.caption('- **Terra**: Powerful model for dense tables and complex documents (Powered by `gpt-4o`).')

        st.radio(
            "AI Model",
            options=["Luna", "Terra"],
            horizontal=True,
            label_visibility="collapsed",
            key="selected_model",
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Row 2: Fields table + Load from CSV ----
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧩 Fields to extract</div>', unsafe_allow_html=True)
    st.caption("Define the fields you wish to extract from each document.")

    col_table, col_csv = st.columns([3, 1])

    with col_table:
        st.data_editor(
            st.session_state.fields_df,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "field": st.column_config.TextColumn(
                    "Field",
                    help="Name of the field to extract",
                ),
                "type": st.column_config.SelectboxColumn(
                    "Type",
                    options=["String", "Number", "Date", "Currency", "Table (markdown string)"],
                    help="Data type of the field",
                ),
                "description": st.column_config.TextColumn(
                    "Description",
                    help="What this field represents",
                ),
            },
            key="fields_editor",
            hide_index=True,
        )

    with col_csv:
        csv_upload = st.file_uploader(
            "📂 Load Fields From CSV",
            type=["csv"],
            key="csv_uploader",
        )
        if csv_upload is not None:
            fingerprint = f"{csv_upload.name}_{csv_upload.size}"
            if st.session_state.get("csv_last_loaded") != fingerprint:
                try:
                    df_loaded = pd.read_csv(csv_upload)
                    df_loaded.columns = [c.strip().lower() for c in df_loaded.columns]
                    for col in ["field", "type", "description"]:
                        if col not in df_loaded.columns:
                            df_loaded[col] = ""
                    st.session_state.fields_df = df_loaded[["field", "type", "description"]].astype(str)
                    if "fields_editor" in st.session_state:
                        del st.session_state["fields_editor"]
                    st.session_state.csv_last_loaded = fingerprint
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load CSV: {e}")

        st.download_button(
            "💾 Save Fields to CSV",
            data=st.session_state.fields_df.to_csv(index=False).encode("utf-8"),
            file_name="fields.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Row 3: Extra instructions table + Load/Save from CSV ----
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">✍ Extra instructions to the AI (Optional)</div>', unsafe_allow_html=True)
    st.caption("Add optional custom instructions to guide AI extraction logic.")

    col_instr, col_instr_csv = st.columns([3, 1])

    with col_instr:
        st.data_editor(
            st.session_state.instructions_df,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "instruction": st.column_config.TextColumn(
                    "Instruction",
                    help="Specific instruction for the AI",
                ),
            },
            key="instructions_editor",
            hide_index=True,
        )

    with col_instr_csv:
        instr_upload = st.file_uploader(
            "📂 Load Instructions From CSV",
            type=["csv"],
            key="instr_csv_uploader",
        )
        if instr_upload is not None:
            fingerprint = f"{instr_upload.name}_{instr_upload.size}"
            if st.session_state.get("instr_csv_last_loaded") != fingerprint:
                try:
                    df_loaded = pd.read_csv(instr_upload)
                    df_loaded.columns = [c.strip().lower() for c in df_loaded.columns]
                    if "instruction" not in df_loaded.columns:
                        df_loaded["instruction"] = ""
                    st.session_state.instructions_df = df_loaded[["instruction"]].astype(str)
                    if "instructions_editor" in st.session_state:
                        del st.session_state["instructions_editor"]
                    st.session_state.instr_csv_last_loaded = fingerprint
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load CSV: {e}")

        st.download_button(
            "💾 Save Instructions to CSV",
            data=st.session_state.instructions_df.to_csv(index=False).encode("utf-8"),
            file_name="instructions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("Next →", type="primary"):
            if not st.session_state.uploaded_files:
                st.toast("Upload some documents first!")
            else:
                st.session_state.step = 2
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# =================================================
# STEP 2 — SUMMARY
# =================================================
elif st.session_state.step == 2:
    if not st.session_state.uploaded_files:
        st.warning("No documents uploaded.")
        if st.button("← Back to Setup"):
            st.session_state.step = 1
            st.rerun()
        st.stop()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Summary</div>', unsafe_allow_html=True)

    docs = len(st.session_state.uploaded_files)
    fields_list = st.session_state.fields_df.to_dict(orient="records")
    num_fields = len([r for r in fields_list if str(r.get("field", "")).strip()])
    
    instructions_list = [r for r in st.session_state.instructions_df.to_dict(orient="records") if str(r.get("instruction", "")).strip()]

    # ── Count total pages across all uploaded PDFs ──────────────────────────
    total_pages = 0
    page_counts = []
    for f in st.session_state.uploaded_files:
        f.seek(0)
        try:
            reader = PdfReader(f)
            n = len(reader.pages)
        except Exception:
            n = 1
        page_counts.append((f.name, n))
        total_pages += n
        f.seek(0)

    # ── Estimates ───────────────────────────────────────────────────────────
    TOKENS_PER_PAGE  = 1200
    PROCESSING_SPEED = 800  # tokens/sec

    est_tokens    = total_pages * TOKENS_PER_PAGE
    est_cost_usd  = get_credits_cost(
        doc_tokens=est_tokens,
        nr_pages=total_pages,
        prompt_tokens=500,
        tokens_output_app=500,
        nr_of_calls=docs,
        model_choice=st.session_state.selected_model
    )

    est_seconds = (est_tokens / PROCESSING_SPEED) + 2
    est_minutes, est_secs = divmod(int(est_seconds), 60)
    time_str = f"{est_minutes}m {est_secs}s" if est_minutes else f"{est_secs}s"

    st.markdown(f"""
**Selected Model:** {st.session_state.selected_model}  
**Documents:** {docs}  
**Total pages:** {total_pages}  
**Fields:** {num_fields}  

⏱️ **Estimated time:** {time_str}  
🔢 **Estimated input tokens:** {est_tokens:,}  
💰 **Estimated cost:** ${est_cost_usd:.5f} USD
""")

    if page_counts:
        st.markdown("**Pages per document**")
        for name, pages in page_counts:
            st.markdown(f"- {name}: **{pages}** page(s)")

    st.markdown("**Fields to extract**")
    for row in fields_list:
        if str(row.get("field", "")).strip():
            st.markdown(f"- **{row['field']}** ({row.get('type', 'String')})")

    if instructions_list:
        st.markdown("**Extra instructions to the AI**")
        for row in instructions_list:
            st.markdown(f"- {row['instruction']}")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Previous"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Start extraction →", type="primary"):
            st.session_state.trigger_extraction = True
            st.session_state.step = 3
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# =================================================
# STEP 3 — RESULTS
# =================================================
elif st.session_state.step == 3:

    if st.session_state.trigger_extraction and st.session_state.uploaded_files:
        
        instructions_strings = [str(r.get("instruction", "")).strip() for r in st.session_state.instructions_df.to_dict(orient="records") if str(r.get("instruction", "")).strip()]
        combined_instructions_str = "\n".join(f"- {inst}" for inst in instructions_strings)
        
        with st.spinner("Running OpenAI extraction…"):
            try:
                result = extract_information_for_documents(
                    files=st.session_state.uploaded_files,
                    fields=st.session_state.fields_df.to_dict(orient="records"),
                    model_choice=st.session_state.selected_model,
                    extra_instructions=combined_instructions_str,
                )
                st.session_state.extraction_output = result
            except Exception as e:
                st.error(f"Extraction Error: {e}")
        st.session_state.trigger_extraction = False

    output = st.session_state.extraction_output
    documents = output.get("documents", []) if output else []

    def _get_value(raw):
        if isinstance(raw, dict):
            return raw.get("value", "—")
        return raw if raw is not None else "—"

    def _get_confidence(raw):
        if isinstance(raw, dict):
            return raw.get("confidence", None)
        return None

    def _get_page(raw):
        if isinstance(raw, dict):
            page = raw.get("page", None)
            return str(page) if page is not None else "—"
        return "—"

    def _get_section(raw):
        if isinstance(raw, dict):
            section = raw.get("section", None)
            return str(section) if section is not None else "—"
        return "—"

    def _confidence_color(conf):
        if conf is None:
            return ""
        if conf >= 0.75:
            return "background-color: #d4edda"
        if conf >= 0.5:
            return "background-color: #fff3cd"
        return "background-color: #f8d7da"

    def _markdown_table_to_df(md: str):
        if not md or not isinstance(md, str):
            return None

        lines = [line.strip() for line in md.strip().splitlines() if line.strip()]
        if len(lines) < 2:
            return None

        table_lines = [line for line in lines if "|" in line]
        if len(table_lines) < 2:
            return None

        def _split_row(line):
            parts = line.strip("| ").split("|")
            return [p.strip() for p in parts]

        headers = None
        data_rows = []

        for line in table_lines:
            cells = _split_row(line)
            if not cells:
                continue

            if all(c in "-|: " for c in line.replace(" ", "")):
                continue

            if headers is None:
                headers = cells
            else:
                data_rows.append(cells)

        if headers is None or len(data_rows) == 0:
            return None

        n = len(headers)
        aligned_rows = [row[:n] + [""] * (n - len(row)) for row in data_rows]

        try:
            return pd.DataFrame(aligned_rows, columns=headers)
        except Exception:
            return None

    st.markdown('<div class="card">', unsafe_allow_html=True)

    hdr_left, hdr_right = st.columns([2, 2])
    with hdr_left:
        st.markdown('<div class="section-title">📊 Results</div>', unsafe_allow_html=True)
        if documents:
            st.caption(f"{len(documents)} document(s) processed with {st.session_state.selected_model}")

    if not documents:
        st.info("No extraction has been run yet. Go back and click **Start extraction →**.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        field_names = [
            str(r.get("field", "")).strip()
            for r in st.session_state.fields_df.to_dict(orient="records")
            if str(r.get("field", "")).strip()
        ]

        table_rows = []
        confidence_rows = []
        for doc in documents:
            if doc["metadata"].get("error"):
                st.error(f"❌ {doc['metadata']['source_file']}: {doc['metadata']['error']}")
            meta = doc.get("metadata", {})
            extracted = doc.get("extracted_fields", {})
            doc_name = meta.get("source_file", meta.get("filename", meta.get("name", "—")))
            row = {"📄 Document": doc_name}
            conf_row = {"📄 Document": None}
            for fn in field_names:
                raw = extracted.get(fn)
                row[fn] = _get_value(raw)
                conf_row[fn] = _get_confidence(raw)
            table_rows.append(row)
            confidence_rows.append(conf_row)

        overview_df = pd.DataFrame(table_rows)
        conf_df = pd.DataFrame(confidence_rows)

        with hdr_right:
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            json_text = json.dumps(output, indent=2)
            with dl_col1:
                st.download_button("⬇️ JSON", json_text, file_name="results.json", mime="application/json", use_container_width=True)
            csv_bytes = overview_df.to_csv(index=False).encode("utf-8")
            with dl_col2:
                st.download_button("⬇️ CSV", csv_bytes, file_name="results.csv", mime="text/csv", use_container_width=True)
            _xlsx_buf = io.BytesIO()
            with pd.ExcelWriter(_xlsx_buf, engine="openpyxl") as _writer:
                overview_df.to_excel(_writer, index=False, sheet_name="Results")
            _xlsx_buf.seek(0)
            with dl_col3:
                st.download_button("⬇️ Excel", _xlsx_buf.read(), file_name="results.xlsx",
                                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                 use_container_width=True)

        st.markdown('<div class="section-title">🗂 Overview</div>', unsafe_allow_html=True)

        def _style_overview(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for col in field_names:
                if col in df.columns:
                    styles[col] = conf_df[col].map(_confidence_color)
            return styles

        styled = overview_df.style.apply(_style_overview, axis=None)
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Per-document details
    if documents:
        st.markdown('<div class="section-title">📄 Per-document detail</div>', unsafe_allow_html=True)

        for i, doc in enumerate(documents):
            meta = doc.get("metadata", {})
            extracted = doc.get("extracted_fields", {})
            doc_name = meta.get("source_file", meta.get("filename", meta.get("name", f"Document_{i + 1}")))

            with st.expander(f"📄 {doc_name}", expanded=(i == 0)):

                viz_key = f"viz_field_doc_{i}"
                if viz_key not in st.session_state:
                    st.session_state[viz_key] = None

                st.markdown("**Extracted fields**")

                if extracted:
                    h1, h2, h3, h4, h5, h6 = st.columns([2, 3.5, 1, 1, 1.5, 1.2])
                    h1.markdown("**Field**")
                    h2.markdown("**Value**")
                    h3.markdown("**Confidence**")
                    h4.markdown("**Page number**")
                    h5.markdown("**Section**")
                    h6.markdown("**Visualize**")

                    st.divider()

                    for k, v in extracted.items():
                        conf = _get_confidence(v)
                        conf_str = f"{int(round(conf * 100))}%" if conf is not None else "—"
                        raw_value = _get_value(v)
                        page_str = _get_page(v)
                        section_str = _get_section(v)

                        val_str = str(raw_value).strip()

                        c1, c2, c3, c4, c5, c6 = st.columns([2, 3.5, 1, 1, 1.5, 1.2])

                        c1.markdown(k)

                        display_val = val_str[:400] + "..." if len(val_str) > 400 else val_str
                        c2.text(display_val)

                        c3.markdown(conf_str)
                        c4.markdown(page_str)
                        c5.markdown(section_str)

                        if c6.button("🗂 Visualize Content", key=f"tv_{i}_{k}", use_container_width=True):
                            st.session_state[viz_key] = (k, raw_value)

                else:
                    st.caption("No fields extracted.")

                st.markdown("---")
                st.markdown("**📊 Visualization**")

                selected = st.session_state.get(viz_key)
                if selected is None:
                    st.caption("Click **Table View** on any field above to see full content or rendered table.")
                else:
                    sel_field, sel_val = selected
                    st.caption(f"Showing full content for: **{sel_field}**")

                    df_vis = _markdown_table_to_df(str(sel_val))

                    if df_vis is not None and not df_vis.empty:
                        st.dataframe(df_vis, use_container_width=True, hide_index=True)
                        
                        pdf_file_name = doc_name.rsplit('.', 1)[0] if '.' in doc_name else doc_name
                        
                        st.download_button(
                            "⬇️ Download CSV",
                            data=df_vis.to_csv(index=False).encode("utf-8"),
                            file_name=f"{pdf_file_name}_{sel_field}.csv",
                            mime="text/csv",
                            key=f"dl_viz_{i}_{sel_field}",
                        )
                    else:
                        st.markdown("**Raw content:**")
                        st.caption(str(sel_val))

    # Navigation
    st.markdown('<div class="card">', unsafe_allow_html=True)
    nav_col1, nav_col2 = st.columns([1, 1])
    with nav_col1:
        if st.button("← Previous"):
            st.session_state.step = 2
            st.rerun()
    with nav_col2:
        if st.button("🔄 Back to Setup"):
            st.session_state.step = 1
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)