# Import necessary libraries
import re
from pdfminer.high_level import extract_text
import sentence_transformers

# Extract text from pdf
def extract_text_from_pdf(pdf_path):
    raw_text = extract_text(pdf_path)
    formatted_text = re.sub(r'[^A-Za-z0-9]+', ' ', raw_text)
    return formatted_text

# Encode CV text
def get_embeddings(text):
    model = sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(text, show_progress_bar=True)
    return embeddings


