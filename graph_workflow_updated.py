import os
import glob
import pandas as pd
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from typing import Any, Dict, Optional, TypedDict
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import END, START, StateGraph
from models.schemas import (
    ClassificationResult,
    ContractQueryResult,
    InvoiceData,
    PurchaseOrderData,
)

load_dotenv()

# Initialize OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

# Use OpenRouter's exact slug for Nemotron Ultra or the automatic free router
MODEL_NAME = "nvidia/nemotron-3-ultra-550b-a55b:free"


class DocumentState(TypedDict):
    pdf_path: str
    user_query: Optional[str]
    extracted_text: str
    extraction_method: str
    uploaded_file: Optional[Any]
    document_type: Optional[str]
    extracted_data: Optional[Any]


# --- DIRECT LLM CALL HELPER ---
def call_llm(prompt_text):
    """Calls the model via OpenRouter's unified endpoint."""
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert document extraction and classification assistant. You must return valid JSON matching the requested keys precisely."},
            {"role": "user", "content": prompt_text}
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    return completion.choices[0].message.content


# --- STEP 1: Smart Ingestion & Text Extraction ---
def ingest_and_extract_node(state: DocumentState) -> Dict[str, Any]:
    path = state["pdf_path"]
    print(f"\n📄 [Smart Ingestion] Inspecting document: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at: {path}")

    extracted_text = ""
    extraction_method = "pdfplumber (selectable)"

    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
    except Exception as e:
        print(f"⚠️ pdfplumber failed: {e}")

    if len(extracted_text.strip()) < 50:
        print("🔍 [OCR Mode] Scanned document detected. Running Tesseract OCR...")
        extraction_method = "pytesseract (scanned/OCR)"
        try:
            images = convert_from_path(path)
            for i, image in enumerate(images):
                ocr_text = pytesseract.image_to_string(image)
                extracted_text += f"--- Page {i + 1} ---\n" + ocr_text + "\n"
        except Exception as e:
            print(f"❌ OCR processing failed: {e}")

    return {
        "extracted_text": extracted_text,
        "extraction_method": extraction_method,
    }


# --- STEP 2: Hybrid Classification (Keyword + LLM) ---
def keyword_classifier(text: str) -> Optional[str]:
    text_lower = text.lower()
    invoice_keywords = ["invoice", "bill to", "tax invoice", "amount due", "invoice no"]
    po_keywords = ["purchase order", "po number", "ship to", "buyer", "vendor code"]
    contract_keywords = ["agreement", "contract", "parties", "hereby", "terms and conditions", "effective date"]

    inv_score = sum(1 for kw in invoice_keywords if kw in text_lower)
    po_score = sum(1 for kw in po_keywords if kw in text_lower)
    con_score = sum(1 for kw in contract_keywords if kw in text_lower)

    max_score = max(inv_score, po_score, con_score)
    if max_score == 0:
        return None

    if max_score == inv_score:
        return "invoice"
    elif max_score == po_score:
        return "purchase_order"
    else:
        return "contract_agreement"


def orchestrator_classifier_node(state: DocumentState) -> Dict[str, Any]:
    print("🧭 [Orchestrator Agent] Running hybrid classification...")
    extracted_text = state["extracted_text"]

    doc_type = keyword_classifier(extracted_text)
    if doc_type:
        print(f"🔑 [Keyword Match] Classified as: **{doc_type.upper()}**")
        return {"document_type": doc_type}

    print("🤖 [LLM Classifier] Querying Classifier...")
    prompt = f"""
    Examine the following extracted document text and classify its category. 
    You must classify it as precisely one of: 'invoice', 'purchase_order', or 'contract_agreement'.
    Return a JSON object with a single key 'document_type'.

    Document Text Excerpt:
    {extracted_text[:3000]}
    """

    raw_json_text = call_llm(prompt)
    result = ClassificationResult.model_validate_json(raw_json_text)
    print(f"🎯 [Orchestrator Decision] Identified type as: **{result.document_type.upper()}**")

    return {"document_type": result.document_type.lower()}


# --- STEP 3A: Invoice Agent ---
def invoice_extractor_node(state: DocumentState) -> Dict[str, Any]:
    print("🧾 [Invoice Agent] Processing invoice data...")
    extracted_text = state["extracted_text"]
    user_query = state.get("user_query")

    query_instruction = f"\nSpecific User Request to address: '{user_query}'" if user_query else ""
    prompt = f"""
    You are an expert Invoice Extractor Agent. Extract information into a JSON object with these exact top-level keys:
    - invoice_number (string)
    - vendor_name (string)
    - customer_name (string)
    - invoice_date (string)
    - due_date (string)
    - total_amount (float)
    - line_items (list of objects, each containing description, quantity, unit_price, total_price)

    Document Text:
    {extracted_text}
    {query_instruction}
    """

    raw_json_text = call_llm(prompt)
    return {"extracted_data": InvoiceData.model_validate_json(raw_json_text)}


# --- STEP 3B: Purchase Order Agent ---
def purchase_order_extractor_node(state: DocumentState) -> Dict[str, Any]:
    print("📦 [Purchase Order Agent] Processing purchase order data...")
    extracted_text = state["extracted_text"]
    user_query = state.get("user_query")

    query_instruction = f"\nSpecific User Request to address: '{user_query}'" if user_query else ""
    prompt = f"""
    You are an expert Purchase Order Extractor Agent. Extract information into a JSON object with these exact top-level keys:
    - po_number (string)
    - buyer_company (string)
    - supplier_name (string)
    - order_date (string)
    - total_order_value (float)
    - line_items (list of objects, each containing item_number, description, quantity, unit_price, total_price)

    Document Text:
    {extracted_text}
    {query_instruction}
    """

    raw_json_text = call_llm(prompt)
    return {"extracted_data": PurchaseOrderData.model_validate_json(raw_json_text)}


# --- STEP 3C: Contract Agent ---
def contract_analyzer_node(state: DocumentState) -> Dict[str, Any]:
    print("⚖️ [Contract Agent] Analyzing legal agreement...")
    extracted_text = state["extracted_text"]
    user_query = state.get("user_query")

    query_instruction = f"\nSpecific User Request / Question to answer: '{user_query}'" if user_query else ""
    prompt = f"""
    You are a Legal Contract Analysis Agent. Review this contract agreement and return a JSON object matching the required schema keys.

    Document Text:
    {extracted_text}
    {query_instruction}
    """

    raw_json_text = call_llm(prompt)
    return {"extracted_data": ContractQueryResult.model_validate_json(raw_json_text)}


def route_to_specialist(state: DocumentState) -> str:
    doc_type = state.get("document_type")
    if doc_type == "invoice":
        return "invoice_agent"
    elif doc_type == "purchase_order":
        return "purchase_order_agent"
    else:
        return "contract_agent"


def create_document_router_graph():
    workflow = StateGraph(DocumentState)

    workflow.add_node("ingest", ingest_and_extract_node)
    workflow.add_node("orchestrator", orchestrator_classifier_node)
    workflow.add_node("invoice_agent", invoice_extractor_node)
    workflow.add_node("purchase_order_agent", purchase_order_extractor_node)
    workflow.add_node("contract_agent", contract_analyzer_node)

    workflow.add_edge(START, "ingest")
    workflow.add_edge("ingest", "orchestrator")

    workflow.add_conditional_edges(
        "orchestrator",
        route_to_specialist,
        {
            "invoice_agent": "invoice_agent",
            "purchase_order_agent": "purchase_order_agent",
            "contract_agent": "contract_agent",
        },
    )

    workflow.add_edge("invoice_agent", END)
    workflow.add_edge("purchase_order_agent", END)
    workflow.add_edge("contract_agent", END)

    return workflow.compile()