import json
import pypdf

from typing import List, Optional

def load_pdf(file_object, num_pages: Optional[int] = None):
    pdf = pypdf.PdfReader(file_object)
    if num_pages is not None:
        return [
        page.extract_text() for page in pdf.pages[:num_pages]
    ]
    return [
        page.extract_text() for page in pdf.pages
    ]

def save_json(chunks_obj: List[str], path: str):
    with open(path, "w") as f:
        json.dump(chunks_obj, f)

def load_json(path: str):
    with open(path, "r") as f:
        chunks = json.load(f)
    return chunks