import streamlit as st
from utils.pdf_processor import PDFProcessor
from utils.text_preprocessor import TextPreprocessor
from utils.financial_extractor import FinancialExtractor
import os

def main():
    st.title("AI Stock Research Analyst")
    st.write("Upload company documents for analysis")
    
    # Initialize processors
    pdf_processor = PDFProcessor()
    text_processor = TextPreprocessor()
    financial_extractor = FinancialExtractor()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Create a temporary directory if it doesn't exist
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save uploaded file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        try:
            # Process the PDF
            with st.spinner('Processing PDF...'):
                # Extract text and tables
                extracted_text = pdf_processor.extract_text(temp_path)
                metadata = pdf_processor.get_metadata()
                tables = pdf_processor.extract_tables(temp_path)
                
                # Process text
                processed_text = text_processor.process_text(extracted_text)
                
                # Extract financial metrics
                financial_metrics = financial_extractor.extract_metrics(extracted_text)
                trends = financial_extractor.analyze_trends(financial_metrics)
            
            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Document Info", "Processed Text", "Financial Metrics", "Tables"])
            
            with tab1:
                st.subheader("Document Metadata")
                st.json(metadata)
                
                st.subheader("Raw Text")
                st.text_area("Content", extracted_text, height=200)
            
            with tab2:
                st.subheader("Processed Text Analysis")
                st.write("Number of sentences:", len(processed_text['sentences']))
                st.write("Number of tokens:", len(processed_text['tokens']))
                st.write("Number of unique tokens (excluding stop words):", 
                         len(set(processed_text['tokens_no_stop'])))
                
                st.subheader("Sample Processed Sentences")
                for i, sent in enumerate(processed_text['sentences'][:5]):
                    st.write(f"{i+1}. {sent}")
            
            with tab3:
                st.subheader("Financial Metrics")
                for metric, data in financial_metrics.items():
                    st.write(f"\n**{metric.title()}**")
                    for mention in data:
                        st.write(f"- Context: {mention['context']}")
                        st.write(f"  Values: {mention['values']}")
                
                st.subheader("Trends Analysis")
                st.json(trends)
            
            with tab4:
                if tables:
                    st.subheader("Extracted Tables")
                    for i, table in enumerate(tables):
                        st.write(f"Table {i+1}")
                        st.table(table)
                else:
                    st.write("No tables found in the document")
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
        
        # Clean up
        os.remove(temp_path)

if __name__ == "__main__":
    main()