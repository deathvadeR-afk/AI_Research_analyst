


          
# AI Stock Research Analyst

An AI-powered application that analyzes company documents (annual reports, earnings calls, etc.) and stock data to provide comprehensive financial analysis and stock performance predictions.

## Features

- PDF document analysis
- Financial metrics extraction
- Sentiment analysis
- Technical analysis
- Stock performance prediction
- Interactive visualization

## Project Structure

```plaintext
.
├── requirements.txt    # Project dependencies
└── src/
    ├── analysis/      # Directory for analysis modules
    ├── data/          # Directory for data storage
    ├── models/        # Machine learning models
    │   └── stock_predictor.py    # Stock prediction model
    ├── reports/       # Generated reports
    ├── utils/         # Utility modules
    │   ├── financial_extractor.py   # Extracts financial metrics
    │   ├── pdf_processor.py         # Processes PDF documents
    │   ├── sentiment_analyzer.py     # Analyzes text sentiment
    │   ├── stock_data_fetcher.py    # Fetches stock market data
    │   └── text_preprocessor.py     # Preprocesses text data
    └── app.py         # Main Streamlit application
```

## Module Descriptions

- **app.py**: Main application file that creates the Streamlit interface and orchestrates all components
- **stock_predictor.py**: Implements machine learning models for stock prediction
- **financial_extractor.py**: Extracts financial metrics from text using regex patterns
- **pdf_processor.py**: Handles PDF document processing and text extraction
- **sentiment_analyzer.py**: Performs sentiment analysis on financial texts
- **stock_data_fetcher.py**: Fetches historical stock data and calculates technical indicators
- **text_preprocessor.py**: Preprocesses text data for analysis

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/deathvadeR-afk/AI_Research_analyst.git
cd AI_Research_analyst
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run src/app.py
```

2. Access the application in your web browser (typically http://localhost:8501)

3. Using the application:
   - Enter a stock ticker symbol in the sidebar (e.g., "AAPL")
   - Upload a PDF document (annual report, earnings call transcript, etc.)
   - Navigate through the tabs to view different analyses:
     - **Document Info**: View document metadata and raw text
     - **Processed Text**: See text analysis results
     - **Financial Metrics**: View extracted financial metrics and trends
     - **Sentiment Analysis**: See sentiment analysis results with visualizations
     - **Tables**: View extracted tables from the document
     - **Stock Prediction**: View stock predictions and analysis

## How It Works

1. **Document Processing**:
   - Uploads PDF document
   - Extracts text and tables using PyPDF2 and pdfplumber
   - Preprocesses text for analysis

2. **Financial Analysis**:
   - Extracts key financial metrics using regex patterns
   - Identifies trends in financial data
   - Analyzes sentiment of financial statements

3. **Stock Analysis**:
   - Fetches historical stock data using yfinance
   - Calculates technical indicators (Moving Averages, RSI, MACD)
   - Combines technical analysis with document analysis

4. **Prediction**:
   - Uses machine learning models to predict stock performance
   - Combines technical indicators, financial metrics, and sentiment
   - Provides confidence scores and contributing factors

## Visualization

The application provides various visualizations:
- Stock price charts with technical indicators
- Sentiment analysis charts
- Feature importance plots
- Financial metrics trends

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK for natural language processing
- yfinance for stock data
- Streamlit for the web interface
- scikit-learn for machine learning capabilities
        