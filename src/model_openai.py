"""
Backend for the PDF field-extraction Streamlit app using OpenAI API.

Pipeline for each PDF:
  1. Extract raw text from the uploaded PDF using PyPDF.
  2. Format the prompt with field definitions and instructions.
  3. Send request to OpenAI API (Luna / gpt-4o-mini or Terra / gpt-4o) using JSON response format.
  4. Parse and structure the extracted result.
"""

import json
import io
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

from prompt import EXTRACTION_PROMPT

# Map user-friendly model names to OpenAI API model identifiers
MODEL_MAPPING = {
    "LUNA": "gpt-4o-mini",
    "TERRA": "gpt-4o",
}

VALID_MODELS = set(MODEL_MAPPING.keys())


def get_openai_client() -> OpenAI:
    """Retrieve OpenAI API Key from Streamlit secrets or environment variable."""
    api_key = None
    try:
        api_key = st.secrets["OPENAI_KEY"]
    except Exception:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_KEY not found in Streamlit secrets (`st.secrets['OPENAI_KEY']`) or environment variables.")
    
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# PDF Text Extraction
# ---------------------------------------------------------------------------

def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract plain text page-by-page from PDF bytes."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    extracted_pages = []
    
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        extracted_pages.append(f"--- PAGE {i + 1} ---\n{page_text}")
        
    return "\n\n".join(extracted_pages)


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------

def _build_extraction_prompt(fields: List[Dict[str, Any]], file_name: str) -> str:
    """Turn the field definitions into prompt specifications."""
    field_specs = []
    for f in fields:
        field_name = f.get("field") or f.get("name")
        if not field_name or not str(field_name).strip():
            continue
        desc = f.get("description") or ""
        ftype = f.get("type") or "String"
        field_specs.append(f'- "{field_name}" (type: {ftype}) — {desc}'.strip())

    if not field_specs:
        raise ValueError(f"No valid fields defined. Current fields = {fields}")

    field_list = "\n".join(field_specs)
    return EXTRACTION_PROMPT.format(file_name=file_name, field_list=field_list)


def _build_text_extraction_prompt(
    fields: List[Dict[str, Any]],
    file_name: str,
    document_text: str,
    extra_instructions: Optional[str] = None,
) -> str:
    """Build the final prompt including extra instructions and inlined PDF text."""
    prompt = _build_extraction_prompt(fields, file_name)

    if extra_instructions and extra_instructions.strip():
        prompt = f"{prompt.rstrip()}\n\nExtra instructions from user:\n{extra_instructions.strip()}\n"

    return f"{prompt.rstrip()}\n\nDOCUMENT TEXT START\n{document_text}\nDOCUMENT TEXT END"


# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON string safely, removing markdown fences if present."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    
    parsed = json.loads(s)
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    return parsed


# ---------------------------------------------------------------------------
# OpenAI Single PDF Extraction
# ---------------------------------------------------------------------------

def _call_openai_for_single_pdf(
    pdf_bytes: bytes,
    file_name: str,
    fields: List[Dict[str, Any]],
    model_choice: str,
    extra_instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract structured fields using OpenAI Chat Completions API."""
    if not pdf_bytes:
        raise ValueError(f"Input document '{file_name}' is empty (0 bytes). Re-upload and try again.")

    client = get_openai_client()
    document_text = _extract_text_from_pdf(pdf_bytes)
    
    prompt = _build_text_extraction_prompt(fields, file_name, document_text, extra_instructions)
    openai_model = MODEL_MAPPING.get(model_choice.upper(), "gpt-4o-mini")

    response = client.chat.completions.create(
        model=openai_model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise data extraction assistant. Output strictly valid JSON with an 'extracted_fields' key mapping each field name to its extracted value, confidence (0.0 to 1.0), page number, and section."
            },
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    raw_content = response.choices[0].message.content
    parsed = _parse_json_response(raw_content)

    if isinstance(parsed, dict) and "extracted_fields" in parsed:
        return {"extracted_fields": parsed["extracted_fields"]}
    elif isinstance(parsed, dict):
        return {"extracted_fields": parsed}
    else:
        raise ValueError(f"Unexpected JSON format from model response: {raw_content[:500]}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_information_for_documents(
    files: List[Any],
    fields: List[Dict[str, Any]],
    model_choice: str = "Luna",
    extra_instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level extraction helper used by Streamlit.

    Args:
        files: Streamlit UploadedFile objects (PDFs).
        fields: List of field definitions to extract.
        model_choice: "Luna" (Default, gpt-4o-mini) or "Terra" (gpt-4o).
        extra_instructions: Custom user instructions.

    Returns:
        Structured JSON dictionary containing per-document extraction outputs.
    """
    selected_model_key = model_choice.upper()
    if selected_model_key not in VALID_MODELS:
        selected_model_key = "LUNA"

    documents: List[Dict[str, Any]] = []

    for uploaded in files:
        file_name = getattr(uploaded, "name", "document.pdf")
        if hasattr(uploaded, "seek"):
            uploaded.seek(0)
        pdf_bytes = uploaded.read()

        document_id = str(uuid.uuid4())
        processed_at = datetime.now(timezone.utc).isoformat()
        error_message: Optional[str] = None

        try:
            llm_result = _call_openai_for_single_pdf(
                pdf_bytes=pdf_bytes,
                file_name=file_name,
                fields=fields,
                model_choice=selected_model_key,
                extra_instructions=extra_instructions,
            )
            status = "success"
            extracted_fields = llm_result["extracted_fields"]
        except Exception as e:
            status = "error"
            extracted_fields = {}
            error_message = str(e)

        metadata: Dict[str, Any] = {
            "document_id": document_id,
            "source_file": file_name,
            "processed_at": processed_at,
            "model_used": f"OpenAI {model_choice} ({MODEL_MAPPING[selected_model_key]})",
            "status": status,
        }
        if error_message:
            metadata["error"] = error_message

        documents.append({"metadata": metadata, "extracted_fields": extracted_fields})

    return {"documents": documents}