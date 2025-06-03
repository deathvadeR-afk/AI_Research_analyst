import PyPDF2
import pdfplumber
from pathlib import Path
from typing import Dict, List

class PDFProcessor:
    def __init__(self):
        self.extracted_text = ""
        self.metadata = {}
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF file using both PyPDF2 and pdfplumber for better accuracy"""
        try:
            text_content = []
            
            # Extract text using PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                self.metadata = {
                    'pages': len(pdf_reader.pages),
                    'author': pdf_reader.metadata.get('/Author', 'Unknown'),
                    'creation_date': pdf_reader.metadata.get('/CreationDate', 'Unknown')
                }
                
                # Use pdfplumber for more accurate text extraction
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text_content.append(page.extract_text() or '')
            
            self.extracted_text = '\n'.join(text_content)
            return self.extracted_text
        
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def get_metadata(self) -> Dict:
        """Return the metadata of the processed PDF"""
        return self.metadata
    
    def extract_tables(self, pdf_path: str) -> List[List[List[str]]]:
        """Extract tables from PDF using pdfplumber"""
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    tables.extend(page.extract_tables() or [])
            return tables
        except Exception as e:
            raise Exception(f"Error extracting tables: {str(e)}")