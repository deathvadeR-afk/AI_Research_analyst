import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class StockDataFetcher:
    def __init__(self):
        self.data_cache = {}
    
    def fetch_stock_data(self, ticker: str, period: str = '1y') -> pd.DataFrame:
        """
        Fetch historical stock data for the given ticker
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            period: Time period to fetch ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            DataFrame with historical stock data
        """
        try:
            # Check if data is already in cache
            cache_key = f"{ticker}_{period}"
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]
            
            # Fetch data from Yahoo Finance
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            # Cache the data
            self.data_cache[cache_key] = data
            
            return data
        except Exception as e:
            raise Exception(f"Error fetching stock data for {ticker}: {str(e)}")
    
    def get_company_info(self, ticker: str) -> Dict:
        """
        Get company information for the given ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info
        except Exception as e:
            raise Exception(f"Error fetching company info for {ticker}: {str(e)}")
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given stock data
        
        Args:
            data: DataFrame with historical stock data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Calculate moving averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD (Moving Average Convergence Divergence)
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        df['20d_std'] = df['Close'].rolling(window=20).std()
        df['upper_band'] = df['MA20'] + (df['20d_std'] * 2)
        df['lower_band'] = df['MA20'] - (df['20d_std'] * 2)
        
        return df