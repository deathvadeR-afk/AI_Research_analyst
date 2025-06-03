import re
import pandas as pd
from typing import List, Dict, Union

class FinancialExtractor:
    def __init__(self):
        self.financial_metrics = {
            'revenue': r'revenue|sales|turnover',
            'profit': r'profit|earnings|net income',
            'margin': r'margin|profit margin|gross margin',
            'growth': r'growth|increase|yoy|year over year',
            'debt': r'debt|liability|loan',
            'assets': r'assets|property|equipment',
            'cash': r'cash|liquidity|cash flow'
        }
    
    def extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text"""
        numbers = re.findall(r'\$?\d+\.?\d*\s?(?:billion|million|thousand|[BMK])?', text)
        processed_numbers = []
        
        for num in numbers:
            try:
                # Remove $ and convert B/M/K to actual numbers
                num = num.replace('$', '').strip()
                if 'billion' in num.lower() or 'B' in num:
                    num = float(re.findall(r'\d+\.?\d*', num)[0]) * 1e9
                elif 'million' in num.lower() or 'M' in num:
                    num = float(re.findall(r'\d+\.?\d*', num)[0]) * 1e6
                elif 'thousand' in num.lower() or 'K' in num:
                    num = float(re.findall(r'\d+\.?\d*', num)[0]) * 1e3
                else:
                    num = float(re.findall(r'\d+\.?\d*', num)[0])
                processed_numbers.append(num)
            except:
                continue
        
        return processed_numbers
    
    def extract_metrics(self, text: str) -> Dict[str, List[str]]:
        """Extract financial metrics and associated values"""
        metrics = {}
        
        for metric, pattern in self.financial_metrics.items():
            # Find sentences containing the metric
            matches = re.finditer(pattern, text.lower())
            metric_mentions = []
            
            for match in matches:
                # Get the sentence containing the metric
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                
                # Extract numbers near the metric
                numbers = self.extract_numbers(context)
                if numbers:
                    metric_mentions.append({
                        'context': context.strip(),
                        'values': numbers
                    })
            
            if metric_mentions:
                metrics[metric] = metric_mentions
        
        return metrics
    
    def analyze_trends(self, metrics: Dict[str, List[str]]) -> Dict[str, str]:
        """Analyze trends in the extracted metrics"""
        trends = {}
        
        for metric, mentions in metrics.items():
            values = []
            for mention in mentions:
                values.extend(mention['values'])
            
            if values:
                trends[metric] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'count': len(values)
                }
        
        return trends