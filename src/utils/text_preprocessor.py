import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Add financial/business specific stop words
        self.stop_words.update(['company', 'business', 'quarter', 'year', 'financial'])
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def tokenize_text(self, text: str) -> list:
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: list) -> list:
        """Remove stopwords from token list"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: list) -> list:
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def extract_sentences(self, text: str) -> list:
        """Extract sentences from text"""
        return sent_tokenize(text)
    
    def process_text(self, text: str) -> dict:
        """Complete text preprocessing pipeline"""
        cleaned_text = self.clean_text(text)
        sentences = self.extract_sentences(cleaned_text)
        tokens = self.tokenize_text(cleaned_text)
        tokens_no_stop = self.remove_stopwords(tokens)
        lemmatized_tokens = self.lemmatize_tokens(tokens_no_stop)
        
        return {
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'tokens': tokens,
            'tokens_no_stop': tokens_no_stop,
            'lemmatized_tokens': lemmatized_tokens
        }