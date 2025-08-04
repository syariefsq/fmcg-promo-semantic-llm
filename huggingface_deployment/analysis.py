"""
NLP Modules for FMCG Promotion Copy Re-use Recommender

This module handles emotion analysis and zero-shot classification for promotional headlines.
It can be called as an autonomous agent or task for orchestration.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    """
    Emotion analysis for promotional headlines using marketing-specific emotions.
    
    This class can be used as an autonomous agent to analyze emotional tones
    in promotional copy.
    """
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        """
        Initialize the EmotionAnalyzer.
        
        Args:
            model_name: Name of the emotion classification model
        """
        self.model_name = model_name
        self.emotion_pipeline = None
        self.marketing_emotions = {
            'joy': ['joy', 'excitement', 'happiness'],
            'urgency': ['fear', 'anger', 'disgust'],  # Map negative emotions to urgency
            'trust': ['neutral', 'surprise'],  # Map neutral/surprise to trust
            'excitement': ['joy', 'surprise'],  # Map joy/surprise to excitement
            'value': ['neutral', 'joy'],  # Map neutral/joy to value perception
            'exclusivity': ['surprise', 'joy']  # Map surprise/joy to exclusivity
        }
        self.is_initialized = False
        
    def initialize(self):
        """
        Initialize the emotion classification pipeline.
        """
        try:
            logger.info(f"Loading emotion classification model: {self.model_name}")
            self.emotion_pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                return_all_scores=True
            )
            self.is_initialized = True
            logger.info("Emotion analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing emotion analyzer: {e}")
            raise
    
    def analyze_emotion(self, text: str) -> Dict:
        """
        Analyze emotion for a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with emotion analysis results
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Get emotion scores
            results = self.emotion_pipeline(text)
            
            # Map to marketing emotions
            marketing_emotion_scores = {}
            for marketing_emotion, base_emotions in self.marketing_emotions.items():
                score = 0
                for base_emotion in base_emotions:
                    for result in results[0]:
                        if result['label'].lower() == base_emotion.lower():
                            score += result['score']
                marketing_emotion_scores[marketing_emotion] = score / len(base_emotions)
            
            # Get primary emotion
            primary_emotion = max(marketing_emotion_scores.items(), key=lambda x: x[1])
            
            return {
                'text': text,
                'primary_emotion': primary_emotion[0],
                'primary_score': primary_emotion[1],
                'all_emotions': marketing_emotion_scores,
                'raw_results': results[0]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing emotion: {e}")
            return {
                'text': text,
                'primary_emotion': 'neutral',
                'primary_score': 1.0,
                'all_emotions': {'neutral': 1.0},
                'error': str(e)
            }
    
    def analyze_batch_emotions(self, texts: List[str]) -> List[Dict]:
        """
        Analyze emotions for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of emotion analysis results
        """
        if not self.is_initialized:
            self.initialize()
        
        logger.info(f"Analyzing emotions for {len(texts)} texts")
        results = []
        
        for text in tqdm(texts, desc="Analyzing emotions"):
            result = self.analyze_emotion(text)
            results.append(result)
        
        return results

class PromoTypeClassifier:
    """
    Zero-shot classification for promotional types.
    
    This class can be used as an autonomous agent to classify promotional
    headlines into different types.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        """
        Initialize the PromoTypeClassifier.
        
        Args:
            model_name: Name of the zero-shot classification model
        """
        self.model_name = model_name
        self.classifier = None
        self.promo_types = [
            "Price-off promotion",
            "Bundle promotion", 
            "Value-add promotion",
            "Sampling promotion",
            "Contest promotion"
        ]
        self.is_initialized = False
        
    def initialize(self):
        """
        Initialize the zero-shot classification pipeline.
        """
        try:
            logger.info(f"Loading zero-shot classification model: {self.model_name}")
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name
            )
            self.is_initialized = True
            logger.info("Promo type classifier initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing promo type classifier: {e}")
            raise
    
    def classify_promo_type(self, text: str) -> Dict:
        """
        Classify promotional type for a single text.
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary with classification results
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Classify
            results = self.classifier(text, self.promo_types)
            
            # Get top prediction
            top_prediction = results['labels'][0]
            top_score = results['scores'][0]
            
            # Map to simplified labels
            type_mapping = {
                "Price-off promotion": "Price-off",
                "Bundle promotion": "Bundle", 
                "Value-add promotion": "Value-add",
                "Sampling promotion": "Sampling",
                "Contest promotion": "Contest"
            }
            
            return {
                'text': text,
                'promo_type': type_mapping.get(top_prediction, top_prediction),
                'confidence': top_score,
                'all_scores': dict(zip([type_mapping.get(label, label) for label in results['labels']], results['scores']))
            }
            
        except Exception as e:
            logger.error(f"Error classifying promo type: {e}")
            return {
                'text': text,
                'promo_type': 'Unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def classify_batch_promo_types(self, texts: List[str]) -> List[Dict]:
        """
        Classify promotional types for a batch of texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of classification results
        """
        if not self.is_initialized:
            self.initialize()
        
        logger.info(f"Classifying promo types for {len(texts)} texts")
        results = []
        
        for text in tqdm(texts, desc="Classifying promo types"):
            result = self.classify_promo_type(text)
            results.append(result)
        
        return results

class NLPAnalyzer:
    """
    Combined NLP analyzer for promotional headlines.
    
    This class combines emotion analysis and promo type classification
    for comprehensive NLP analysis.
    """
    
    def __init__(self):
        """
        Initialize the NLPAnalyzer.
        """
        self.emotion_analyzer = EmotionAnalyzer()
        self.promo_classifier = PromoTypeClassifier()
        self.is_initialized = False
        
    def initialize(self):
        """
        Initialize both NLP components.
        """
        try:
            logger.info("Initializing NLP analyzer components...")
            self.emotion_analyzer.initialize()
            self.promo_classifier.initialize()
            self.is_initialized = True
            logger.info("NLP analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP analyzer: {e}")
            raise
    
    def analyze_headline(self, headline: str) -> Dict:
        """
        Perform complete NLP analysis on a promotional headline.
        
        Args:
            headline: Promotional headline to analyze
            
        Returns:
            Dictionary with complete NLP analysis
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Analyze emotion
            emotion_result = self.emotion_analyzer.analyze_emotion(headline)
            
            # Classify promo type
            type_result = self.promo_classifier.classify_promo_type(headline)
            
            return {
                'headline': headline,
                'emotion_analysis': emotion_result,
                'promo_type_classification': type_result,
                'summary': {
                    'primary_emotion': emotion_result['primary_emotion'],
                    'emotion_score': emotion_result['primary_score'],
                    'promo_type': type_result['promo_type'],
                    'type_confidence': type_result['confidence']
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing headline: {e}")
            return {
                'headline': headline,
                'error': str(e)
            }
    
    def analyze_batch_headlines(self, headlines: List[str]) -> List[Dict]:
        """
        Perform complete NLP analysis on a batch of headlines.
        
        Args:
            headlines: List of promotional headlines to analyze
            
        Returns:
            List of complete NLP analysis results
        """
        if not self.is_initialized:
            self.initialize()
        
        logger.info(f"Analyzing {len(headlines)} headlines with NLP")
        results = []
        
        for headline in tqdm(headlines, desc="Analyzing headlines"):
            result = self.analyze_headline(headline)
            results.append(result)
        
        return results
    
    def analyze_promos_dataframe(self, promos_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze promotional data from a DataFrame and add NLP results.
        
        Args:
            promos_df: DataFrame with promotional data
            
        Returns:
            DataFrame with added NLP analysis columns
        """
        if not self.is_initialized:
            self.initialize()
        
        logger.info(f"Analyzing {len(promos_df)} promotional records")
        
        # Analyze all headlines
        headlines = promos_df['headline'].tolist()
        nlp_results = self.analyze_batch_headlines(headlines)
        
        # Add results to DataFrame
        df = promos_df.copy()
        
        # Extract emotion and type information
        df['primary_emotion'] = [result.get('summary', {}).get('primary_emotion', 'neutral') 
                               for result in nlp_results]
        df['emotion_score'] = [result.get('summary', {}).get('emotion_score', 0.0) 
                             for result in nlp_results]
        df['promo_type'] = [result.get('summary', {}).get('promo_type', 'Unknown') 
                          for result in nlp_results]
        df['type_confidence'] = [result.get('summary', {}).get('type_confidence', 0.0) 
                               for result in nlp_results]
        
        # Add full analysis as JSON column
        df['nlp_analysis'] = nlp_results
        
        logger.info("NLP analysis completed and added to DataFrame")
        return df

# Agent-compatible functions for orchestration
def initialize_nlp_analyzer() -> Dict:
    """
    Agent function to initialize the NLP analyzer.
    
    Returns:
        Dictionary with initialization results
    """
    try:
        analyzer = NLPAnalyzer()
        analyzer.initialize()
        
        return {
            'status': 'success',
            'message': 'NLP analyzer initialized successfully',
            'components': ['emotion_analyzer', 'promo_type_classifier']
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error initializing NLP analyzer: {str(e)}'
        }

def analyze_single_headline(headline: str) -> Dict:
    """
    Agent function to analyze a single promotional headline.
    
    Args:
        headline: Promotional headline to analyze
        
    Returns:
        Dictionary with NLP analysis results
    """
    analyzer = NLPAnalyzer()
    result = analyzer.analyze_headline(headline)
    
    return {
        'status': 'success',
        'headline': headline,
        'analysis': result
    }

def analyze_promos_batch(headlines: List[str]) -> Dict:
    """
    Agent function to analyze a batch of promotional headlines.
    
    Args:
        headlines: List of promotional headlines to analyze
        
    Returns:
        Dictionary with batch analysis results
    """
    analyzer = NLPAnalyzer()
    results = analyzer.analyze_batch_headlines(headlines)
    
    return {
        'status': 'success',
        'headlines_count': len(headlines),
        'results': results
    }

def analyze_promos_dataframe_agent(promos_df: pd.DataFrame) -> Dict:
    """
    Agent function to analyze promotional data from DataFrame.
    
    Args:
        promos_df: DataFrame with promotional data
        
    Returns:
        Dictionary with analysis results and enhanced DataFrame
    """
    analyzer = NLPAnalyzer()
    enhanced_df = analyzer.analyze_promos_dataframe(promos_df)
    
    # Get summary statistics
    emotion_distribution = enhanced_df['primary_emotion'].value_counts().to_dict()
    type_distribution = enhanced_df['promo_type'].value_counts().to_dict()
    
    return {
        'status': 'success',
        'records_analyzed': len(enhanced_df),
        'emotion_distribution': emotion_distribution,
        'type_distribution': type_distribution,
        'enhanced_dataframe': enhanced_df
    }

def get_emotion_analysis(headline: str) -> Dict:
    """
    Agent function to get emotion analysis for a headline.
    
    Args:
        headline: Promotional headline to analyze
        
    Returns:
        Dictionary with emotion analysis results
    """
    analyzer = EmotionAnalyzer()
    result = analyzer.analyze_emotion(headline)
    
    return {
        'status': 'success',
        'headline': headline,
        'emotion_analysis': result
    }

def get_promo_type_classification(headline: str) -> Dict:
    """
    Agent function to get promo type classification for a headline.
    
    Args:
        headline: Promotional headline to classify
        
    Returns:
        Dictionary with classification results
    """
    classifier = PromoTypeClassifier()
    result = classifier.classify_promo_type(headline)
    
    return {
        'status': 'success',
        'headline': headline,
        'classification': result
    }

if __name__ == "__main__":
    # Example usage
    analyzer = NLPAnalyzer()
    analyzer.initialize()
    
    # Test with sample headlines
    test_headlines = [
        "BELI 2 GRATIS 1 - Hemat Rp 15.000!",
        "DISKON 50% - Stok Terbatas!",
        "GRATIS SAMPLING - Coba Sekarang!",
        "WIN PRIZE - Kirim Foto & Menang!",
        "GRATIS TUMBLER - Beli 3 Dapat 1!"
    ]
    
    print("Testing NLP analysis...")
    for headline in test_headlines:
        result = analyzer.analyze_headline(headline)
        print(f"\nHeadline: {headline}")
        print(f"Primary Emotion: {result['summary']['primary_emotion']} ({result['summary']['emotion_score']:.3f})")
        print(f"Promo Type: {result['summary']['promo_type']} ({result['summary']['type_confidence']:.3f})") 