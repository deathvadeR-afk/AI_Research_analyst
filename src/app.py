import streamlit as st
from utils.pdf_processor import PDFProcessor
from utils.text_preprocessor import TextPreprocessor
from utils.financial_extractor import FinancialExtractor
from utils.sentiment_analyzer import FinancialSentimentAnalyzer
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("AI Stock Research Analyst")
    st.write("Upload company documents for analysis")
    
    # Initialize processors
    pdf_processor = PDFProcessor()
    text_processor = TextPreprocessor()
    financial_extractor = FinancialExtractor()
    sentiment_analyzer = FinancialSentimentAnalyzer()
    
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
                
                # Sentiment analysis
                sentences = text_processor.extract_sentences(extracted_text)
                sentiment_summary = sentiment_analyzer.get_sentiment_summary(sentences)
                
                # Topic-based sentiment analysis
                financial_topics = sentiment_analyzer.get_financial_topics()
                topic_sentiments = sentiment_analyzer.analyze_by_topic(sentences, financial_topics)
            
            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Document Info", "Processed Text", "Financial Metrics", "Sentiment Analysis", "Tables"])
            
            with tab1:
                st.subheader("Document Metadata")
                st.json(metadata)
                
                st.subheader("Raw Text")
                st.text_area("Content", extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text, height=200)
            
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
                st.subheader("Overall Sentiment Analysis")
                
                # Display overall sentiment
                overall_sentiment = sentiment_summary['overall_sentiment']
                sentiment_color = "green" if overall_sentiment == "positive" else "red" if overall_sentiment == "negative" else "gray"
                st.markdown(f"<h3 style='color: {sentiment_color}'>Overall Sentiment: {overall_sentiment.title()}</h3>", unsafe_allow_html=True)
                
                # Display sentiment scores
                st.write("Average Sentiment Scores:")
                scores_df = pd.DataFrame({
                    'Score': [sentiment_summary['average_scores']['compound'], 
                              sentiment_summary['average_scores']['positive'],
                              sentiment_summary['average_scores']['negative'],
                              sentiment_summary['average_scores']['neutral']],
                    'Type': ['Compound', 'Positive', 'Negative', 'Neutral']
                })
                
                # Create a bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Type', y='Score', data=scores_df, ax=ax, 
                            palette=['blue', 'green', 'red', 'gray'])
                ax.set_title('Sentiment Scores')
                st.pyplot(fig)
                
                # Display sentence counts
                st.write("Sentence Distribution:")
                counts_df = pd.DataFrame({
                    'Count': [sentiment_summary['sentence_counts']['positive'],
                              sentiment_summary['sentence_counts']['negative'],
                              sentiment_summary['sentence_counts']['neutral']],
                    'Type': ['Positive', 'Negative', 'Neutral']
                })
                
                # Create a pie chart
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(counts_df['Count'], labels=counts_df['Type'], autopct='%1.1f%%',
                       colors=['green', 'red', 'gray'])
                ax.set_title('Sentence Sentiment Distribution')
                st.pyplot(fig)
                
                # Display most positive and negative sentences
                st.subheader("Most Positive Sentence")
                st.write(sentiment_summary['most_positive_sentence'])
                
                st.subheader("Most Negative Sentence")
                st.write(sentiment_summary['most_negative_sentence'])
                
                # Display topic-based sentiment
                st.subheader("Topic-Based Sentiment Analysis")
                
                for topic, sentiment in topic_sentiments.items():
                    st.write(f"\n**{topic.title()}**")
                    st.write(f"Overall: {sentiment['overall_sentiment'].title()}")
                    st.write(f"Sentences: {sentiment['sentence_counts']['total']}")
                    st.write(f"Compound Score: {sentiment['average_scores']['compound']:.2f}")
                    
                    # Display a sample positive and negative sentence for this topic
                    if sentiment['most_positive_sentence']:
                        st.write("Sample Positive: ", sentiment['most_positive_sentence'])
                    if sentiment['most_negative_sentence']:
                        st.write("Sample Negative: ", sentiment['most_negative_sentence'])
            
            with tab5:
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