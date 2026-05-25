import os
from pypdf import PdfReader

def load_annual_report(fiscal_year: str) -> str:
    """
    Dynamically extracts text from actual physical ITC Annual Report PDFs.
    """
    filename_mapping = {
        "FY25": "ITC-Report-and-Accounts-2025.pdf",
        "FY24": "ITC-Report-and-Accounts-2024.pdf",
        "FY23": "ITC-Report-and-Accounts-2023.pdf",
        "FY22": "ITC-Report-and-Accounts-2022.pdf"
    }
    
    filename = filename_mapping.get(fiscal_year.upper())
    if not filename:
        return ""
        
    file_path = os.path.join("data", filename)
    
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: Physical file {file_path} not found.")
        return ""
        
    try:
        print(f"📖 Parsing raw PDF content from: {file_path}...")
        reader = PdfReader(file_path)
        extracted_text = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
                
        return "\n".join(extracted_text)
    except Exception as e:
        print(f"❌ Error parsing {filename}: {str(e)}")
        return ""