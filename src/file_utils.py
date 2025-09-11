import fitz  # PyMuPDF
import docx2txt
import tempfile

def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF file uploaded via Streamlit"""
    text = ""
    # Read uploaded file as bytes
    pdf_bytes = uploaded_file.read()
    # Reset file pointer so it can be reused later if needed
    uploaded_file.seek(0)

    # Open PDF from bytes stream
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text


def extract_text_from_docx(uploaded_file):
    """Extract text from a DOCX file uploaded via Streamlit"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Reset file pointer
    uploaded_file.seek(0)

    # Extract text using docx2txt
    text = docx2txt.process(tmp_path)
    return text
