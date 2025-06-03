import streamlit as st
from utils.pdf_processor import PDFProcessor
from utils.text_preprocessor import TextPreprocessor
from utils.financial_extractor import FinancialExtractor
from utils.sentiment_analyzer import FinancialSentimentAnalyzer
from utils.stock_data_fetcher import StockDataFetcher
from models.stock_predictor import StockPredictor
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    st.title("AI Stock Research Analyst")
    st.write("Upload company documents for analysis")
    
    # Initialize processors
    pdf_processor = PDFProcessor()
    text_processor = TextPreprocessor()
    financial_extractor = FinancialExtractor()
    sentiment_analyzer = FinancialSentimentAnalyzer()
    stock_data_fetcher = StockDataFetcher()
    stock_predictor = StockPredictor()
    
    # Sidebar for stock ticker input
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "")
    
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
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Document Info", "Processed Text", "Financial Metrics", "Sentiment Analysis", "Tables", "Stock Prediction"])
            
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
            
            with tab6:
                st.subheader("Stock Performance Prediction")
                
                if not ticker:
                    st.warning("Please enter a stock ticker in the sidebar to enable stock prediction")
                else:
                    try:
                        with st.spinner(f'Fetching data for {ticker}...'):
                            # Fetch stock data
                            stock_data = stock_data_fetcher.fetch_stock_data(ticker, period='1y')
                            company_info = stock_data_fetcher.get_company_info(ticker)
                            
                            # Calculate technical indicators
                            stock_data_with_indicators = stock_data_fetcher.calculate_technical_indicators(stock_data)
                            
                            # Display company info
                            st.subheader(f"Company: {company_info.get('shortName', ticker)}")
                            st.write(f"Sector: {company_info.get('sector', 'N/A')}")
                            st.write(f"Industry: {company_info.get('industry', 'N/A')}")
                            
                            # Display stock chart
                            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                               vertical_spacing=0.1, 
                                               subplot_titles=('Price', 'Volume'),
                                               row_heights=[0.7, 0.3])
                            
                            # Add price candlestick
                            fig.add_trace(
                                go.Candlestick(
                                    x=stock_data.index,
                                    open=stock_data['Open'],
                                    high=stock_data['High'],
                                    low=stock_data['Low'],
                                    close=stock_data['Close'],
                                    name='Price'
                                ),
                                row=1, col=1
                            )
                            
                            # Add volume bar chart
                            fig.add_trace(
                                go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume'),
                                row=2, col=1
                            )
                            
                            # Add moving averages
                            if 'MA20' in stock_data_with_indicators.columns:
                                fig.add_trace(
                                    go.Scatter(x=stock_data_with_indicators.index, 
                                              y=stock_data_with_indicators['MA20'], 
                                              name='20-day MA',
                                              line=dict(color='blue', width=1)),
                                    row=1, col=1
                                )
                            
                            if 'MA50' in stock_data_with_indicators.columns:
                                fig.add_trace(
                                    go.Scatter(x=stock_data_with_indicators.index, 
                                              y=stock_data_with_indicators['MA50'], 
                                              name='50-day MA',
                                              line=dict(color='orange', width=1)),
                                    row=1, col=1
                                )
                            
                            # Update layout
                            fig.update_layout(
                                title=f'{ticker} Stock Price',
                                xaxis_title='Date',
                                yaxis_title='Price',
                                height=600,
                                width=800,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig)
                            
                            # Prepare features for prediction
                            features = stock_predictor.prepare_features(
                                stock_data_with_indicators, 
                                sentiment_score=sentiment_summary['average_scores']['compound']
                            )
                            
                            # Train model
                            with st.spinner('Training prediction model...'):
                                training_results = stock_predictor.train(features)
                                
                                # Make prediction
                                prediction = stock_predictor.predict(features)
                                
                                # Combine with financial metrics and sentiment
                                recommendation = stock_predictor.evaluate_with_financial_metrics(
                                    prediction, trends, sentiment_summary
                                )
                            
                            # Display prediction results
                            st.subheader("Prediction Results")
                            
                            # Overall recommendation
                            rec_color = "green" if recommendation['overall_signal'] == "up" else "red"
                            st.markdown(f"<h3 style='color: {rec_color}'>{recommendation['summary']}</h3>", unsafe_allow_html=True)
                            
                            # Display factors table
                            st.subheader("Contributing Factors")
                            factors_df = pd.DataFrame(recommendation['factors'])
                            st.table(factors_df[['factor', 'signal', 'description']])
                            
                            # Display model metrics
                            st.subheader("Model Performance")
                            st.write(f"Mean Squared Error: {training_results['mse']:.4f}")
                            st.write(f"Mean Absolute Error: {training_results['mae']:.4f}")
                            st.write(f"R² Score: {training_results['r2']:.4f}")
                            
                            # Display feature importance
                            st.subheader("Feature Importance")
                            importance_df = pd.DataFrame({
                                'Feature': list(training_results['feature_importance'].keys()),
                                'Importance': list(training_results['feature_importance'].values())
                            }).sort_values('Importance', ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax)
                            ax.set_title('Top 10 Feature Importance')
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"Error in stock prediction: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
        
        # Clean up
        os.remove(temp_path)

if __name__ == "__main__":
    main()