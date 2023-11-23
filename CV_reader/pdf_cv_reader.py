# Import necessary libraries
from pdfminer.high_level import extract_text

# Extract text from pdf
def extract_text_from_pdf(pdf_path):
    raw_text = extract_text(pdf_path)
    formatted_text = raw_text.replace('\n', ' ')
    return formatted_text

print(pdfminer.__version__)