import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Dict, List, Tuple

class FinancialSentimentAnalyzer:
    def __init__(self):
        # Initialize VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()
        
        # Add financial-specific terms to the VADER lexicon
        self._add_financial_lexicon()
    
    def _add_financial_lexicon(self):
        """Add financial domain-specific terms to the sentiment lexicon"""
        financial_lexicon = {
            # Positive financial terms
            'growth': 2.0,
            'profit': 2.0,
            'profitable': 2.0,
            'increase': 1.5,
            'increasing': 1.5,
            'exceeded': 1.8,
            'beat': 1.5,
            'above': 1.0,
            'strong': 1.5,
            'strength': 1.5,
            'opportunity': 1.5,
            'opportunities': 1.5,
            'positive': 1.5,
            'improvement': 1.5,
            'improved': 1.5,
            'dividend': 1.0,
            'dividends': 1.0,
            
            # Negative financial terms
            'loss': -2.0,
            'losses': -2.0,
            'debt': -1.5,
            'decrease': -1.5,
            'decreasing': -1.5,
            'declined': -1.5,
            'below': -1.0,
            'weak': -1.5,
            'weakness': -1.5,
            'risk': -1.0,
            'risks': -1.0,
            'negative': -1.5,
            'challenges': -1.0,
            'challenging': -1.0,
            'missed': -1.5,
            'restructuring': -1.0,
            'layoff': -2.0,
            'layoffs': -2.0,
            'litigation': -1.5,
            'lawsuit': -1.5,
            'penalty': -1.5,
            'investigation': -1.0,
        }
        
        # Update the VADER lexicon with our financial terms
        self.vader.lexicon.update(financial_lexicon)
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze the sentiment of the given text"""
        return self.vader.polarity_scores(text)
    
    def analyze_sentence_sentiments(self, sentences: List[str]) -> List[Dict]:
        """Analyze sentiment for each sentence"""
        return [self.analyze_sentiment(sentence) for sentence in sentences]
    
    def get_sentiment_summary(self, sentences: List[str]) -> Dict:
        """Get a summary of sentiment across all sentences"""
        sentiments = self.analyze_sentence_sentiments(sentences)
        
        # Calculate average sentiment scores
        avg_compound = sum(s['compound'] for s in sentiments) / len(sentiments) if sentiments else 0
        avg_pos = sum(s['pos'] for s in sentiments) / len(sentiments) if sentiments else 0
        avg_neg = sum(s['neg'] for s in sentiments) / len(sentiments) if sentiments else 0
        avg_neu = sum(s['neu'] for s in sentiments) / len(sentiments) if sentiments else 0
        
        # Count sentences by sentiment category
        positive_sentences = sum(1 for s in sentiments if s['compound'] >= 0.05)
        negative_sentences = sum(1 for s in sentiments if s['compound'] <= -0.05)
        neutral_sentences = sum(1 for s in sentiments if -0.05 < s['compound'] < 0.05)
        
        # Get most positive and negative sentences
        sorted_sentiments = [(s['compound'], i) for i, s in enumerate(sentiments)]
        most_positive_idx = max(sorted_sentiments, key=lambda x: x[0])[1] if sorted_sentiments else -1
        most_negative_idx = min(sorted_sentiments, key=lambda x: x[0])[1] if sorted_sentiments else -1
        
        most_positive_sentence = sentences[most_positive_idx] if most_positive_idx != -1 else ""
        most_negative_sentence = sentences[most_negative_idx] if most_negative_idx != -1 else ""
        
        return {
            'average_scores': {
                'compound': avg_compound,
                'positive': avg_pos,
                'negative': avg_neg,
                'neutral': avg_neu
            },
            'sentence_counts': {
                'positive': positive_sentences,
                'negative': negative_sentences,
                'neutral': neutral_sentences,
                'total': len(sentences)
            },
            'most_positive_sentence': most_positive_sentence,
            'most_negative_sentence': most_negative_sentence,
            'overall_sentiment': 'positive' if avg_compound >= 0.05 else 'negative' if avg_compound <= -0.05 else 'neutral'
        }
    
    def analyze_by_topic(self, sentences: List[str], topics: Dict[str, List[str]]) -> Dict[str, Dict]:
        """Analyze sentiment grouped by financial topics"""
        topic_sentiments = {}
        
        for topic, keywords in topics.items():
            # Find sentences related to this topic
            topic_sentences = []
            for sentence in sentences:
                if any(keyword.lower() in sentence.lower() for keyword in keywords):
                    topic_sentences.append(sentence)
            
            if topic_sentences:
                # Get sentiment summary for this topic
                topic_sentiments[topic] = self.get_sentiment_summary(topic_sentences)
        
        return topic_sentiments
    
    def get_financial_topics(self) -> Dict[str, List[str]]:
        """Return predefined financial topics and their related keywords"""
        return {
            'revenue': ['revenue', 'sales', 'turnover', 'income', 'earnings'],
            'profit': ['profit', 'margin', 'earnings', 'ebitda', 'net income'],
            'growth': ['growth', 'increase', 'expand', 'growing', 'grew'],
            'cost': ['cost', 'expense', 'spending', 'expenditure'],
            'investment': ['invest', 'investment', 'capital', 'capex', 'r&d'],
            'outlook': ['outlook', 'forecast', 'guidance', 'expect', 'future'],
            'competition': ['competition', 'competitor', 'market share', 'industry'],
            'risk': ['risk', 'uncertainty', 'challenge', 'threat', 'litigation']
        }