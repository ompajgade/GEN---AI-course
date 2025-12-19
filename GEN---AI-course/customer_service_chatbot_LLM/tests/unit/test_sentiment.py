"""
Unit tests for Sentiment Analysis Engine (Task 5)
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from task5_sentiment.sentiment_analysis import SentimentAnalysisEngine, SentimentResult


class TestSentimentResult(unittest.TestCase):
    """Test cases for SentimentResult class."""
    
    def test_sentiment_result_creation(self):
        """Test sentiment result creation."""
        result = SentimentResult(
            label="positive",
            score=0.85,
            raw_scores={"positive": 0.85, "negative": 0.10, "neutral": 0.05}
        )
        
        self.assertEqual(result.label, "positive")
        self.assertEqual(result.score, 0.85)
        self.assertEqual(result.raw_scores["positive"], 0.85)
        self.assertIsNotNone(result.timestamp)
    
    def test_sentiment_result_with_timestamp(self):
        """Test sentiment result with custom timestamp."""
        custom_timestamp = "2023-01-01T00:00:00"
        result = SentimentResult(
            label="negative",
            score=0.75,
            raw_scores={"positive": 0.15, "negative": 0.75, "neutral": 0.10},
            timestamp=custom_timestamp
        )
        
        self.assertEqual(result.timestamp, custom_timestamp)


class TestSentimentAnalysisEngine(unittest.TestCase):
    """Test cases for SentimentAnalysisEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the transformers pipeline to avoid loading actual models
        self.mock_pipeline = Mock()
        
        with patch('task5_sentiment.sentiment_analysis.pipeline') as mock_pipeline_func:
            mock_pipeline_func.return_value = self.mock_pipeline
            with patch('task5_sentiment.sentiment_analysis.AutoTokenizer'):
                with patch('task5_sentiment.sentiment_analysis.AutoModelForSequenceClassification'):
                    self.engine = SentimentAnalysisEngine()
    
    def test_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.model_name, "distilbert-base-uncased-finetuned-sst-2-english")
        self.assertIn('positive', self.engine.sentiment_thresholds)
        self.assertIn('negative', self.engine.sentiment_thresholds)
        self.assertIn('positive', self.engine.tone_adjustments)
    
    def test_analyze_sentiment_positive(self):
        """Test sentiment analysis for positive text."""
        # Mock pipeline response
        self.mock_pipeline.return_value = [[
            {'label': 'POSITIVE', 'score': 0.85},
            {'label': 'NEGATIVE', 'score': 0.15}
        ]]
        
        result = self.engine.analyze_sentiment("I love this product!")
        
        self.assertIsInstance(result, SentimentResult)
        self.assertEqual(result.label, "positive")
        self.assertEqual(result.score, 0.85)
        self.assertIn("positive", result.raw_scores)
        self.assertIn("negative", result.raw_scores)
    
    def test_analyze_sentiment_negative(self):
        """Test sentiment analysis for negative text."""
        # Mock pipeline response
        self.mock_pipeline.return_value = [[
            {'label': 'NEGATIVE', 'score': 0.80},
            {'label': 'POSITIVE', 'score': 0.20}
        ]]
        
        result = self.engine.analyze_sentiment("This is terrible!")
        
        self.assertEqual(result.label, "negative")
        self.assertEqual(result.score, 0.80)
    
    def test_analyze_sentiment_neutral_low_confidence(self):
        """Test sentiment analysis with low confidence (should be neutral)."""
        # Mock pipeline response with low confidence
        self.mock_pipeline.return_value = [[
            {'label': 'POSITIVE', 'score': 0.55},
            {'label': 'NEGATIVE', 'score': 0.45}
        ]]
        
        result = self.engine.analyze_sentiment("The weather is okay.")
        
        # Should be classified as neutral due to low confidence
        self.assertEqual(result.label, "neutral")
    
    def test_analyze_sentiment_empty_text(self):
        """Test sentiment analysis with empty text."""
        result = self.engine.analyze_sentiment("")
        
        self.assertEqual(result.label, "neutral")
        self.assertEqual(result.score, 0.5)
        self.assertIn("positive", result.raw_scores)
        self.assertIn("negative", result.raw_scores)
        self.assertIn("neutral", result.raw_scores)
    
    def test_analyze_sentiment_whitespace_only(self):
        """Test sentiment analysis with whitespace-only text."""
        result = self.engine.analyze_sentiment("   \n\t   ")
        
        self.assertEqual(result.label, "neutral")
        self.assertEqual(result.score, 0.5)
    
    def test_analyze_sentiment_error(self):
        """Test sentiment analysis with pipeline error."""
        # Mock pipeline to raise exception
        self.mock_pipeline.side_effect = Exception("Pipeline error")
        
        result = self.engine.analyze_sentiment("Test text")
        
        # Should return neutral sentiment as fallback
        self.assertEqual(result.label, "neutral")
        self.assertEqual(result.score, 0.5)
    
    def test_get_sentiment_label(self):
        """Test getting just the sentiment label."""
        # Mock pipeline response
        self.mock_pipeline.return_value = [[
            {'label': 'POSITIVE', 'score': 0.85},
            {'label': 'NEGATIVE', 'score': 0.15}
        ]]
        
        label = self.engine.get_sentiment_label("I love this!")
        
        self.assertEqual(label, "positive")
    
    def test_get_sentiment_score(self):
        """Test getting just the sentiment score."""
        # Mock pipeline response
        self.mock_pipeline.return_value = [[
            {'label': 'NEGATIVE', 'score': 0.75},
            {'label': 'POSITIVE', 'score': 0.25}
        ]]
        
        score = self.engine.get_sentiment_score("This is bad!")
        
        self.assertEqual(score, 0.75)
    
    def test_adjust_response_tone_positive(self):
        """Test response tone adjustment for positive sentiment."""
        original = "Here is your information."
        adjusted = self.engine.adjust_response_tone(original, "positive")
        
        self.assertIn("I'm glad to help!", adjusted)
        self.assertIn("I hope this helps!", adjusted)
        self.assertIn(original, adjusted)
    
    def test_adjust_response_tone_negative(self):
        """Test response tone adjustment for negative sentiment."""
        original = "Here is your information."
        adjusted = self.engine.adjust_response_tone(original, "negative")
        
        self.assertIn("I understand this might be frustrating", adjusted)
        self.assertIn("I'm here to help you through this", adjusted)
        self.assertIn(original, adjusted)
    
    def test_adjust_response_tone_neutral(self):
        """Test response tone adjustment for neutral sentiment."""
        original = "Here is your information."
        adjusted = self.engine.adjust_response_tone(original, "neutral")
        
        self.assertIn("Let me know if you need any clarification", adjusted)
        self.assertIn(original, adjusted)
    
    def test_adjust_response_tone_invalid_sentiment(self):
        """Test response tone adjustment with invalid sentiment."""
        original = "Here is your information."
        adjusted = self.engine.adjust_response_tone(original, "invalid")
        
        # Should default to neutral
        self.assertIn("Let me know if you need any clarification", adjusted)
    
    def test_make_empathetic(self):
        """Test making text more empathetic."""
        text = "This can't be done and it's wrong."
        empathetic = self.engine._make_empathetic(text)
        
        self.assertIn("might not be able to", empathetic)
        self.assertIn("not quite right", empathetic)
        self.assertNotIn("can't", empathetic)
        self.assertNotIn("wrong", empathetic)
    
    def test_make_enthusiastic(self):
        """Test making text more enthusiastic."""
        text = "That's good and yes, it's fine."
        enthusiastic = self.engine._make_enthusiastic(text)
        
        self.assertIn("excellent", enthusiastic)
        self.assertIn("absolutely", enthusiastic)
        self.assertIn("wonderful", enthusiastic)
    
    def test_generate_empathetic_response_negative(self):
        """Test generating empathetic response for negative sentiment."""
        response = self.engine.generate_empathetic_response("I'm having issues", "negative")
        
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        # Should contain supportive language
        self.assertTrue(any(word in response.lower() for word in 
                          ["understand", "difficult", "help", "support", "solution"]))
    
    def test_generate_empathetic_response_positive(self):
        """Test generating empathetic response for positive sentiment."""
        response = self.engine.generate_empathetic_response("This is great!", "positive")
        
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        # Should contain enthusiastic language
        self.assertTrue(any(word in response.lower() for word in 
                          ["glad", "excited", "wonderful", "happy", "great"]))
    
    def test_generate_empathetic_response_neutral(self):
        """Test generating empathetic response for neutral sentiment."""
        response = self.engine.generate_empathetic_response("I need information", "neutral")
        
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        # Should contain professional language
        self.assertTrue(any(word in response.lower() for word in 
                          ["thank", "understand", "help", "assist", "provide"]))
    
    def test_batch_analyze(self):
        """Test batch sentiment analysis."""
        texts = ["I love this!", "This is terrible!", "It's okay."]
        
        # Mock pipeline responses
        self.mock_pipeline.side_effect = [
            [[{'label': 'POSITIVE', 'score': 0.85}, {'label': 'NEGATIVE', 'score': 0.15}]],
            [[{'label': 'NEGATIVE', 'score': 0.80}, {'label': 'POSITIVE', 'score': 0.20}]],
            [[{'label': 'POSITIVE', 'score': 0.55}, {'label': 'NEGATIVE', 'score': 0.45}]]
        ]
        
        results = self.engine.batch_analyze(texts)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].label, "positive")
        self.assertEqual(results[1].label, "negative")
        self.assertEqual(results[2].label, "neutral")  # Low confidence
    
    def test_get_sentiment_statistics(self):
        """Test getting sentiment statistics."""
        texts = ["I love this!", "This is terrible!", "It's okay.", "Great job!"]
        
        # Mock pipeline responses
        self.mock_pipeline.side_effect = [
            [[{'label': 'POSITIVE', 'score': 0.85}, {'label': 'NEGATIVE', 'score': 0.15}]],
            [[{'label': 'NEGATIVE', 'score': 0.80}, {'label': 'POSITIVE', 'score': 0.20}]],
            [[{'label': 'POSITIVE', 'score': 0.55}, {'label': 'NEGATIVE', 'score': 0.45}]],
            [[{'label': 'POSITIVE', 'score': 0.90}, {'label': 'NEGATIVE', 'score': 0.10}]]
        ]
        
        stats = self.engine.get_sentiment_statistics(texts)
        
        self.assertEqual(stats['total_texts'], 4)
        self.assertIn('sentiment_distribution', stats)
        self.assertIn('average_scores', stats)
        self.assertIn('most_common_sentiment', stats)
        
        # Check that distributions sum to 1
        distribution = stats['sentiment_distribution']
        total_distribution = sum(distribution.values())
        self.assertAlmostEqual(total_distribution, 1.0, places=2)
    
    def test_update_thresholds(self):
        """Test updating sentiment thresholds."""
        new_thresholds = {
            'positive': 0.8,
            'negative': 0.7
        }
        
        self.engine.update_thresholds(new_thresholds)
        
        self.assertEqual(self.engine.sentiment_thresholds['positive'], 0.8)
        self.assertEqual(self.engine.sentiment_thresholds['negative'], 0.7)


class TestSentimentAnalysisEngineIntegration(unittest.TestCase):
    """Integration tests for SentimentAnalysisEngine."""
    
    @patch('task5_sentiment.sentiment_analysis.TRANSFORMERS_AVAILABLE', False)
    def test_initialization_without_transformers(self):
        """Test initialization when transformers is not available."""
        with self.assertRaises(ImportError):
            SentimentAnalysisEngine()
    
    def test_sentiment_thresholds_configuration(self):
        """Test that sentiment thresholds are properly configured."""
        with patch('task5_sentiment.sentiment_analysis.pipeline'):
            with patch('task5_sentiment.sentiment_analysis.AutoTokenizer'):
                with patch('task5_sentiment.sentiment_analysis.AutoModelForSequenceClassification'):
                    engine = SentimentAnalysisEngine()
        
        self.assertIn('positive', engine.sentiment_thresholds)
        self.assertIn('negative', engine.sentiment_thresholds)
        self.assertIn('neutral_min', engine.sentiment_thresholds)
        self.assertIn('neutral_max', engine.sentiment_thresholds)
        
        # Check reasonable threshold values
        self.assertGreater(engine.sentiment_thresholds['positive'], 0.5)
        self.assertGreater(engine.sentiment_thresholds['negative'], 0.3)
    
    def test_tone_adjustments_configuration(self):
        """Test that tone adjustments are properly configured."""
        with patch('task5_sentiment.sentiment_analysis.pipeline'):
            with patch('task5_sentiment.sentiment_analysis.AutoTokenizer'):
                with patch('task5_sentiment.sentiment_analysis.AutoModelForSequenceClassification'):
                    engine = SentimentAnalysisEngine()
        
        for sentiment in ['positive', 'negative', 'neutral']:
            self.assertIn(sentiment, engine.tone_adjustments)
            self.assertIn('prefix', engine.tone_adjustments[sentiment])
            self.assertIn('style', engine.tone_adjustments[sentiment])
            self.assertIn('closing', engine.tone_adjustments[sentiment])


if __name__ == '__main__':
    unittest.main()