import os
import io
import pandas as pd
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path
import boto3

# Initialize the Textract client
# Initialize the Textract client
textract_client = boto3.client('textract', region_name='ap-south-1')  # Change to your AWS region


def is_valid_invoice_field(key, val):
    """Filters out noise, empty values, line items, and footers, keeping only valid invoice metadata."""
    if not val or val.lower() == 'none' or val.strip() == '':
        return False

    key_upper = key.upper()

    # Define allowed keywords that belong to invoice headers/summary sections
    allowed_keywords = [
        'INVOICE', 'P.O.', 'PO NUMBER', 'DATE', 'TERM',
        'SALESPERSON', 'TO', 'SHIP', 'SUBTOTAL', 'TAX',
        'TOTAL', 'SHIPPING', 'DUE', 'BALANCE'
    ]

    has_keyword = any(kw in key_upper for kw in allowed_keywords)
    is_short_header = len(key_upper) <= 35

    return has_keyword and is_short_header


def extract_form_fields(page_bytes):
    """Extracts and filters key-value form fields using analyze_document."""
    response = textract_client.analyze_document(
        Document={'Bytes': page_bytes},
        FeatureTypes=['FORMS']
    )

    blocks = response.get('Blocks', [])
    block_map = {block['Id']: block for block in blocks}
    form_fields = {}

    for block in blocks:
        if block['BlockType'] == 'KEY_VALUE_SET' and 'KEY' in block['EntityTypes']:
            key_text = ""
            if 'Relationships' in block:
                for rel in block['Relationships']:
                    if rel['Type'] == 'CHILD':
                        for cid in rel['Ids']:
                            if block_map[cid]['BlockType'] == 'WORD':
                                key_text += block_map[cid]['Text'] + " "

            val_text = ""
            if 'Relationships' in block:
                for rel in block['Relationships']:
                    if rel['Type'] == 'VALUE':
                        for vid in rel['Ids']:
                            val_block = block_map[vid]
                            if 'Relationships' in val_block:
                                for v_rel in val_block['Relationships']:
                                    if v_rel['Type'] == 'CHILD':
                                        for child_id in v_rel['Ids']:
                                            if block_map[child_id]['BlockType'] == 'WORD':
                                                val_text += block_map[child_id]['Text'] + " "

            clean_key = key_text.strip()
            clean_val = val_text.strip()

            if is_valid_invoice_field(clean_key, clean_val):
                form_fields[clean_key] = clean_val

    return form_fields


def extract_tabular_data(page_bytes):
    """Extracts line items / tabular data using analyze_expense."""
    response = textract_client.analyze_expense(
        Document={'Bytes': page_bytes}
    )

    line_items = []
    for expense_doc in response.get('ExpenseDocuments', []):
        for item_group in expense_doc.get('LineItemGroups', []):
            for line_item in item_group.get('LineItems', []):
                row_data = {}
                for expense_field in line_item.get('LineItemExpenseFields', []):
                    col_type = expense_field.get('Type', {}).get('Text', 'UNKNOWN')
                    col_value = expense_field.get('ValueDetection', {}).get('Text', '')
                    row_data[col_type] = col_value
                line_items.append(row_data)

    return line_items


def is_pdf_selectable(file_path):
    """Determines whether a PDF contains selectable digital text or is scanned."""
    try:
        reader = PdfReader(file_path)
        total_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                total_text += text.strip()
        # If text length is substantial, it's digital/selectable. Otherwise, it's scanned.
        return len(total_text) > 50
    except Exception:
        return False


def process_file(local_path):
    """Handles selectable PDFs, scanned PDFs, and standalone images."""
    combined_forms = {}
    combined_line_items = []

    file_extension = os.path.splitext(local_path)[1].lower()

    if file_extension == '.pdf':
        # Check if the PDF is selectable or scanned
        selectable = is_pdf_selectable(local_path)

        if selectable:
            print(" -> PDF Type: Selectable (Digital text found)")
            reader = PdfReader(local_path)
            total_pages = len(reader.pages)
            print(f"Processing selectable PDF with {total_pages} page(s)...")

            for index, page in enumerate(reader.pages):
                print(f"   -> Processing page {index + 1} of {total_pages}...")
                writer = PdfWriter()
                writer.add_page(page)
                byte_stream = io.BytesIO()
                writer.write(byte_stream)
                page_bytes = byte_stream.getvalue()

                page_forms = extract_form_fields(page_bytes)
                page_items = extract_tabular_data(page_bytes)

                combined_forms.update(page_forms)
                combined_line_items.extend(page_items)
        else:
            print(" -> PDF Type: Scanned (Converting pages to images via pdf2image)")
            # Convert scanned PDF pages to high-res PIL Images
            images = convert_from_path(local_path, dpi=200)
            total_pages = len(images)
            print(f"Processing scanned PDF with {total_pages} page(s)...")

            for index, pil_image in enumerate(images):
                print(f"   -> Processing page image {index + 1} of {total_pages}...")
                byte_stream = io.BytesIO()
                pil_image.save(byte_stream, format='PNG')
                page_bytes = byte_stream.getvalue()

                page_forms = extract_form_fields(page_bytes)
                page_items = extract_tabular_data(page_bytes)

                combined_forms.update(page_forms)
                combined_line_items.extend(page_items)

    elif file_extension in ['.png', '.jpg', '.jpeg']:
        print(" -> File Type: Image")
        with open(local_path, 'rb') as f:
            page_bytes = f.read()

        combined_forms = extract_form_fields(page_bytes)
        combined_line_items = extract_tabular_data(page_bytes)

    return {
        "FormFields": combined_forms,
        "LineItems": combined_line_items
    }


def save_invoice_to_excel(filename, result, output_folder_path):
    """Creates a folder named after the file inside output_folder_path and saves the excel file."""
    file_base_name = os.path.splitext(filename)[0]
    file_output_dir = os.path.join(output_folder_path, file_base_name)
    os.makedirs(file_output_dir, exist_ok=True)

    excel_path = os.path.join(file_output_dir, "extracted_invoice_data.xlsx")

    form_records = [{"File Name": filename, **result["FormFields"]}]
    line_item_records = [{"File Name": filename, **item} for item in result["LineItems"]]

    df_forms = pd.DataFrame(form_records)
    df_items = pd.DataFrame(line_item_records)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_forms.to_excel(writer, sheet_name='Header Form Fields', index=False)
        df_items.to_excel(writer, sheet_name='Tabular Line Items', index=False)

    print(f"Successfully saved Excel file inside folder: '{file_output_dir}'")


def main(input_path, output_folder_path):
    """Main function taking input file/directory path and output folder path."""
    if not os.path.exists(input_path):
        print(f"Input path '{input_path}' does not exist.")
        return

    os.makedirs(output_folder_path, exist_ok=True)

    files_to_process = []
    if os.path.isfile(input_path):
        files_to_process = [input_path]
    elif os.path.isdir(input_path):
        files_to_process = [
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))
        ]

    if not files_to_process:
        print("No valid invoice files found to process.")
        return

    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        print(f"\n==========================================")
        print(f"Processing File: {filename}")
        print(f"==========================================")

        try:
            result = process_file(file_path)
            save_invoice_to_excel(filename, result, output_folder_path)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    INPUT_PATH = './invoices'
    OUTPUT_FOLDER = './output_results'

    main(INPUT_PATH, OUTPUT_FOLDER)