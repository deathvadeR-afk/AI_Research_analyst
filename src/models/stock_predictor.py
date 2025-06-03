import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class StockPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
    
    def prepare_features(self, stock_data: pd.DataFrame, sentiment_score: float = None) -> pd.DataFrame:
        """
        Prepare features for the prediction model
        
        Args:
            stock_data: DataFrame with historical stock data and technical indicators
            sentiment_score: Optional sentiment score to include as a feature
            
        Returns:
            DataFrame with features for prediction
        """
        # Drop rows with NaN values (from calculating indicators)
        df = stock_data.dropna().copy()
        
        # Create features
        features = pd.DataFrame()
        
        # Technical indicators
        features['ma20'] = df['MA20']
        features['ma50'] = df['MA50']
        features['ma200'] = df['MA200']
        features['rsi'] = df['RSI']
        features['macd'] = df['MACD']
        features['signal'] = df['Signal']
        features['upper_band'] = df['upper_band']
        features['lower_band'] = df['lower_band']
        
        # Price-based features
        features['close'] = df['Close']
        features['volume'] = df['Volume']
        features['high_low_diff'] = df['High'] - df['Low']
        features['close_open_diff'] = df['Close'] - df['Open']
        
        # Trend features
        features['price_change'] = df['Close'].pct_change()
        features['volume_change'] = df['Volume'].pct_change()
        
        # Add sentiment score if provided
        if sentiment_score is not None:
            features['sentiment'] = sentiment_score
        
        # Drop rows with NaN values from feature engineering
        features = features.dropna()
        
        return features
    
    def train(self, features: pd.DataFrame, target_days: int = 5) -> Dict:
        """
        Train the prediction model
        
        Args:
            features: DataFrame with features
            target_days: Number of days ahead to predict
            
        Returns:
            Dictionary with training results
        """
        # Create target variable (future price change)
        y = features['close'].shift(-target_days) / features['close'] - 1
        
        # Remove rows with NaN in target
        features = features.iloc[:-target_days]
        y = y.iloc[:-target_days]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate feature importance
        self.feature_importance = dict(zip(features.columns, self.model.feature_importances_))
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'feature_importance': self.feature_importance
        }
    
    def predict(self, features: pd.DataFrame) -> Dict:
        """
        Make predictions using the trained model
        
        Args:
            features: DataFrame with features
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise Exception("Model not trained yet")
        
        # Scale features
        features_scaled = self.scaler.transform(features.iloc[-1:].values)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        # Determine prediction direction
        direction = "up" if prediction > 0 else "down"
        confidence = abs(prediction)
        
        return {
            'predicted_change': prediction,
            'direction': direction,
            'confidence': confidence,
            'top_factors': sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def evaluate_with_financial_metrics(self, 
                                       prediction: Dict, 
                                       financial_metrics: Dict, 
                                       sentiment_summary: Dict) -> Dict:
        """
        Combine model prediction with financial metrics and sentiment for a comprehensive evaluation
        
        Args:
            prediction: Dictionary with model prediction
            financial_metrics: Dictionary with extracted financial metrics
            sentiment_summary: Dictionary with sentiment analysis results
            
        Returns:
            Dictionary with comprehensive evaluation
        """
        # Extract key metrics
        revenue_trend = self._extract_trend(financial_metrics, 'revenue')
        profit_trend = self._extract_trend(financial_metrics, 'profit')
        growth_trend = self._extract_trend(financial_metrics, 'growth')
        
        # Get sentiment scores
        sentiment_score = sentiment_summary['average_scores']['compound']
        sentiment_category = sentiment_summary['overall_sentiment']
        
        # Combine all factors
        factors = []
        
        # Add model prediction
        factors.append({
            'factor': 'Technical Analysis',
            'signal': prediction['direction'],
            'strength': prediction['confidence'],
            'description': f"Technical indicators suggest stock will go {prediction['direction']} with {prediction['confidence']:.2%} confidence"
        })
        
        # Add financial metrics
        if revenue_trend:
            factors.append({
                'factor': 'Revenue',
                'signal': 'up' if revenue_trend['avg'] > 0 else 'down',
                'strength': abs(revenue_trend['avg']) / 100,
                'description': f"Revenue trend is {revenue_trend['avg']:.2f}% based on {revenue_trend['count']} mentions"
            })
        
        if profit_trend:
            factors.append({
                'factor': 'Profit',
                'signal': 'up' if profit_trend['avg'] > 0 else 'down',
                'strength': abs(profit_trend['avg']) / 100,
                'description': f"Profit trend is {profit_trend['avg']:.2f}% based on {profit_trend['count']} mentions"
            })
        
        if growth_trend:
            factors.append({
                'factor': 'Growth',
                'signal': 'up' if growth_trend['avg'] > 0 else 'down',
                'strength': abs(growth_trend['avg']) / 100,
                'description': f"Growth trend is {growth_trend['avg']:.2f}% based on {growth_trend['count']} mentions"
            })
        
        # Add sentiment
        factors.append({
            'factor': 'Sentiment',
            'signal': 'up' if sentiment_score > 0 else 'down',
            'strength': abs(sentiment_score),
            'description': f"Document sentiment is {sentiment_category} with score {sentiment_score:.2f}"
        })
        
        # Calculate overall recommendation
        up_signals = sum(1 for f in factors if f['signal'] == 'up')
        down_signals = sum(1 for f in factors if f['signal'] == 'down')
        
        # Weight by strength
        weighted_signal = sum(f['strength'] if f['signal'] == 'up' else -f['strength'] for f in factors)
        
        recommendation = {
            'overall_signal': 'up' if weighted_signal > 0 else 'down',
            'confidence': min(abs(weighted_signal), 1.0),  # Cap at 1.0
            'factors': factors,
            'summary': f"Based on {len(factors)} factors, the overall recommendation is {'BUY' if weighted_signal > 0 else 'SELL'} with {min(abs(weighted_signal), 1.0):.2%} confidence"
        }
        
        return recommendation
    
    def _extract_trend(self, financial_metrics: Dict, metric_name: str) -> Optional[Dict]:
        """
        Extract trend information for a specific financial metric
        
        Args:
            financial_metrics: Dictionary with extracted financial metrics
            metric_name: Name of the metric to extract trend for
            
        Returns:
            Dictionary with trend information or None if not available
        """
        if metric_name not in financial_metrics:
            return None
        
        return financial_metrics.get(metric_name, {})