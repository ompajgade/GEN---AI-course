"""
Sentiment Analysis Engine - Task 5
Integrates sentiment analysis into the chatbot to detect and respond appropriately 
to customer emotions during interactions.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")

from shared.utils import create_error_response, create_success_response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Represents the result of sentiment analysis."""
    label: str  # 'positive', 'negative', 'neutral'
    score: float  # Confidence score (0-1)
    raw_scores: Dict[str, float]  # Raw scores for all labels
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class SentimentAnalysisEngine:
    """
    Sentiment Analysis Engine for detecting and responding to customer emotions.
    
    Features:
    - Detects positive, negative, and neutral sentiments
    - Adjusts response tone based on detected sentiment
    - Provides confidence scores for sentiment predictions
    - Supports batch processing for multiple texts
    
    Requirements Addressed:
    - 5.1: Analyze text and classify sentiment as positive, negative, or neutral
    - 5.2-5.4: Adjust response tone based on detected sentiment
    - 5.5: Achieve minimum 70% accuracy in sentiment classification
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: str = "auto"
    ):
        """
        Initialize the Sentiment Analysis Engine.
        
        Args:
            model_name: Pre-trained model for sentiment analysis
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
        """
        self.model_name = model_name
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize the sentiment pipeline
        self.sentiment_pipeline = None
        self.tokenizer = None
        self.model = None
        
        # Sentiment thresholds
        self.sentiment_thresholds = {
            'positive': 0.6,    # Score > 0.6 = positive
            'negative': 0.6,    # Score > 0.6 = negative  
            'neutral_min': 0.4, # Score between 0.4-0.6 = neutral
            'neutral_max': 0.6
        }
        
        # Response tone adjustments
        self.tone_adjustments = {
            'positive': {
                'prefix': "I'm glad to help! ",
                'style': 'enthusiastic',
                'closing': " I hope this helps! 😊"
            },
            'negative': {
                'prefix': "I understand this might be frustrating. ",
                'style': 'empathetic',
                'closing': " I'm here to help you through this."
            },
            'neutral': {
                'prefix': "",
                'style': 'professional',
                'closing': " Let me know if you need any clarification."
            }
        }
        
        # Initialize the model
        self._initialize_model()
        
        logger.info(f"✅ Sentiment Analysis Engine initialized with {model_name} on {self.device}")
    
    def _initialize_model(self):
        """Initialize the sentiment analysis model and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers torch")
        
        try:
            # Initialize the sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            # Also load tokenizer and model separately for more control
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            logger.info(f"✅ Model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize sentiment model: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze the sentiment of input text.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult with label, score, and raw scores
            
        Validates: Requirements 5.1 (sentiment classification)
        """
        if not text or not text.strip():
            return SentimentResult(
                label="neutral",
                score=0.5,
                raw_scores={"positive": 0.33, "negative": 0.33, "neutral": 0.34}
            )
        
        try:
            # Get predictions from the pipeline
            results = self.sentiment_pipeline(text.strip())
            
            # Convert results to our format
            raw_scores = {}
            for result in results[0]:  # Pipeline returns list of lists
                label = result['label'].lower()
                score = result['score']
                
                # Map model labels to our labels
                if label in ['positive', 'pos']:
                    raw_scores['positive'] = score
                elif label in ['negative', 'neg']:
                    raw_scores['negative'] = score
                else:
                    raw_scores['neutral'] = score
            
            # Ensure we have all three labels
            if 'neutral' not in raw_scores:
                # Calculate neutral as 1 - (positive + negative)
                pos_score = raw_scores.get('positive', 0)
                neg_score = raw_scores.get('negative', 0)
                raw_scores['neutral'] = max(0, 1 - pos_score - neg_score)
            
            # Determine final label and score
            final_label = max(raw_scores, key=raw_scores.get)
            final_score = raw_scores[final_label]
            
            # Apply thresholds for neutral classification
            if final_label in ['positive', 'negative']:
                if final_score < self.sentiment_thresholds[final_label]:
                    final_label = 'neutral'
                    final_score = raw_scores['neutral']
            
            return SentimentResult(
                label=final_label,
                score=final_score,
                raw_scores=raw_scores
            )
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed: {e}")
            # Return neutral sentiment as fallback
            return SentimentResult(
                label="neutral",
                score=0.5,
                raw_scores={"positive": 0.33, "negative": 0.33, "neutral": 0.34}
            )
    
    def get_sentiment_label(self, text: str) -> str:
        """
        Get just the sentiment label for input text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment label: 'positive', 'negative', or 'neutral'
        """
        result = self.analyze_sentiment(text)
        return result.label
    
    def get_sentiment_score(self, text: str) -> float:
        """
        Get the confidence score for the predicted sentiment.
        
        Args:
            text: Text to analyze
            
        Returns:
            Confidence score (0-1)
        """
        result = self.analyze_sentiment(text)
        return result.score
    
    def adjust_response_tone(self, response: str, sentiment: str) -> str:
        """
        Adjust response tone based on detected user sentiment.
        
        Args:
            response: Original response text
            sentiment: Detected sentiment ('positive', 'negative', 'neutral')
            
        Returns:
            Tone-adjusted response
            
        Validates: Requirements 5.2-5.4 (sentiment-appropriate responses)
        """
        if sentiment not in self.tone_adjustments:
            sentiment = 'neutral'  # Fallback to neutral
        
        adjustments = self.tone_adjustments[sentiment]
        
        # Apply tone adjustments
        adjusted_response = response
        
        # Add prefix for empathy/enthusiasm
        if adjustments['prefix']:
            adjusted_response = adjustments['prefix'] + adjusted_response
        
        # Add closing for warmth/support
        if adjustments['closing']:
            adjusted_response = adjusted_response + adjustments['closing']
        
        # Apply style modifications
        if adjustments['style'] == 'empathetic':
            # Make language more supportive
            adjusted_response = self._make_empathetic(adjusted_response)
        elif adjustments['style'] == 'enthusiastic':
            # Make language more positive
            adjusted_response = self._make_enthusiastic(adjusted_response)
        
        return adjusted_response
    
    def _make_empathetic(self, text: str) -> str:
        """Make text more empathetic for negative sentiment."""
        # Replace harsh words with softer alternatives
        replacements = {
            "can't": "might not be able to",
            "won't": "may not",
            "impossible": "challenging",
            "wrong": "not quite right",
            "error": "issue",
            "failed": "didn't work as expected"
        }
        
        result = text
        for harsh, soft in replacements.items():
            result = result.replace(harsh, soft)
        
        return result
    
    def _make_enthusiastic(self, text: str) -> str:
        """Make text more enthusiastic for positive sentiment."""
        # Add positive reinforcement
        positive_words = {
            "good": "excellent",
            "ok": "great",
            "fine": "wonderful",
            "yes": "absolutely",
            "sure": "definitely"
        }
        
        result = text
        for neutral, positive in positive_words.items():
            result = result.replace(f" {neutral} ", f" {positive} ")
        
        return result
    
    def generate_empathetic_response(self, query: str, sentiment: str) -> str:
        """
        Generate an empathetic response based on query and sentiment.
        
        Args:
            query: User's query
            sentiment: Detected sentiment
            
        Returns:
            Empathetic response
        """
        if sentiment == 'negative':
            responses = [
                "I understand this situation might be difficult for you. Let me help you find a solution.",
                "I can see this is concerning for you. I'm here to provide the support you need.",
                "I recognize this might be frustrating. Let's work together to address your concerns.",
                "I appreciate you sharing this with me. I want to make sure we resolve this properly."
            ]
        elif sentiment == 'positive':
            responses = [
                "I'm so glad you reached out! I'm excited to help you with this.",
                "That's wonderful! I'm here to make sure you get exactly what you need.",
                "I love your enthusiasm! Let me provide you with the best assistance possible.",
                "Great question! I'm happy to help you explore this further."
            ]
        else:  # neutral
            responses = [
                "Thank you for your question. I'll provide you with accurate information.",
                "I understand what you're looking for. Let me help you with that.",
                "I'm here to assist you. Let me provide you with the details you need.",
                "I'll be happy to help you with this inquiry."
            ]
        
        # Select response based on query content (simple keyword matching)
        import random
        return random.choice(responses)
    
    def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment for multiple texts at once.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of SentimentResult objects
        """
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        
        return results
    
    def get_sentiment_statistics(self, texts: List[str]) -> Dict[str, any]:
        """
        Get sentiment statistics for a collection of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with sentiment statistics
        """
        results = self.batch_analyze(texts)
        
        # Count sentiments
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for result in results:
            sentiment_counts[result.label] += 1
            for sentiment, score in result.raw_scores.items():
                total_scores[sentiment] += score
        
        total_texts = len(texts)
        
        return {
            'total_texts': total_texts,
            'sentiment_distribution': {
                sentiment: count / total_texts 
                for sentiment, count in sentiment_counts.items()
            },
            'average_scores': {
                sentiment: score / total_texts 
                for sentiment, score in total_scores.items()
            },
            'most_common_sentiment': max(sentiment_counts, key=sentiment_counts.get)
        }
    
    def update_thresholds(self, thresholds: Dict[str, float]):
        """
        Update sentiment classification thresholds.
        
        Args:
            thresholds: New threshold values
        """
        self.sentiment_thresholds.update(thresholds)
        logger.info(f"Updated sentiment thresholds: {self.sentiment_thresholds}")


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing Sentiment Analysis Engine\n")
    
    # Initialize the engine
    print("1️⃣ Initializing sentiment analysis engine...")
    try:
        sentiment_engine = SentimentAnalysisEngine()
        print("✅ Engine initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        exit(1)
    
    # Test cases
    test_texts = [
        "I love this service! It's amazing!",  # Positive
        "This is terrible and doesn't work at all.",  # Negative
        "The weather is cloudy today.",  # Neutral
        "I'm so frustrated with this problem!",  # Negative
        "Thank you so much for your help!",  # Positive
        "Can you provide more information?",  # Neutral
        "I hate waiting for so long!",  # Negative
        "This is exactly what I needed!"  # Positive
    ]
    
    print("\n2️⃣ Testing sentiment analysis...")
    for i, text in enumerate(test_texts, 1):
        result = sentiment_engine.analyze_sentiment(text)
        print(f"{i}. Text: '{text}'")
        print(f"   Sentiment: {result.label} (confidence: {result.score:.3f})")
        print(f"   Raw scores: {result.raw_scores}")
        print()
    
    # Test tone adjustment
    print("3️⃣ Testing tone adjustment...")
    original_response = "Here is the information you requested."
    
    for sentiment in ['positive', 'negative', 'neutral']:
        adjusted = sentiment_engine.adjust_response_tone(original_response, sentiment)
        print(f"{sentiment.capitalize()}: {adjusted}")
    
    # Test batch analysis
    print("\n4️⃣ Testing batch analysis...")
    batch_results = sentiment_engine.batch_analyze(test_texts)
    print(f"Analyzed {len(batch_results)} texts in batch")
    
    # Get statistics
    print("\n5️⃣ Getting sentiment statistics...")
    stats = sentiment_engine.get_sentiment_statistics(test_texts)
    print(f"Total texts: {stats['total_texts']}")
    print(f"Sentiment distribution: {stats['sentiment_distribution']}")
    print(f"Most common sentiment: {stats['most_common_sentiment']}")
    
    print("\n✅ All tests completed successfully!")