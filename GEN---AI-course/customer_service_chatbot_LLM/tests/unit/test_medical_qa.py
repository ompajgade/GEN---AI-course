"""
Unit tests for Medical Q&A System (Task 3)
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from task3_medical_qa.medical_qa import MedicalQASystem


class TestMedicalQASystem(unittest.TestCase):
    """Test cases for MedicalQASystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies to avoid actual database operations
        self.mock_vector_db = Mock()
        self.mock_entity_recognizer = Mock()
        self.mock_embedding_service = Mock()
        
        # Initialize system with mocked dependencies
        self.qa_system = MedicalQASystem(
            vector_db=self.mock_vector_db,
            entity_recognizer=self.mock_entity_recognizer,
            embedding_service=self.mock_embedding_service,
            enable_sentiment=False,
            enable_multilingual=False
        )
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.qa_system)
        self.assertEqual(self.qa_system.collection_name, "medical_knowledge")
        self.assertFalse(self.qa_system.is_loaded)
        self.assertEqual(len(self.qa_system.qa_pairs), 0)
    
    def test_check_existing_data_empty(self):
        """Test checking existing data when empty."""
        # Mock empty collection
        self.mock_vector_db.get_collection_stats.return_value = {'document_count': 0}
        
        self.qa_system._check_existing_data()
        self.assertFalse(self.qa_system.is_loaded)
    
    def test_check_existing_data_loaded(self):
        """Test checking existing data when loaded."""
        # Mock loaded collection
        self.mock_vector_db.get_collection_stats.return_value = {'document_count': 100}
        
        self.qa_system._check_existing_data()
        self.assertTrue(self.qa_system.is_loaded)
    
    def test_process_medical_query_not_loaded(self):
        """Test processing query when dataset not loaded."""
        result = self.qa_system.process_medical_query("What is diabetes?")
        
        self.assertIn('error', result)
        self.assertIn('Dataset not loaded', result['error'])
        self.assertEqual(result['query'], "What is diabetes?")
        self.assertEqual(result['entities'], {})
        self.assertEqual(result['answers'], [])
    
    @patch('task3_medical_qa.medical_qa.MedicalQASystem._check_existing_data')
    def test_process_medical_query_success(self, mock_check):
        """Test successful medical query processing."""
        # Set up system as loaded
        self.qa_system.is_loaded = True
        
        # Mock entity recognition
        self.mock_entity_recognizer.extract_medical_entities.return_value = {
            'diseases': [Mock(text='diabetes', entity_type='disease', confidence=0.9)]
        }
        
        # Mock embedding generation
        self.mock_embedding_service.generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock vector database query
        mock_metadata = {
            'question': 'What is diabetes?',
            'answer': 'Diabetes is a metabolic disorder.',
            'source': 'Test Source',
            'url': 'http://test.com'
        }
        self.qa_system.retrieve_relevant_answers = Mock(return_value={
            'metadatas': [mock_metadata],
            'distances': [0.1]
        })
        
        result = self.qa_system.process_medical_query("What is diabetes?")
        
        self.assertEqual(result['query'], "What is diabetes?")
        self.assertIn('entities', result)
        self.assertIn('answers', result)
        self.assertEqual(len(result['answers']), 1)
        self.assertEqual(result['answers'][0]['answer'], 'Diabetes is a metabolic disorder.')
    
    def test_extract_medical_entities(self):
        """Test medical entity extraction."""
        # Mock entity recognizer response
        mock_entity = Mock()
        mock_entity.text = 'diabetes'
        mock_entity.entity_type = 'disease'
        mock_entity.confidence = 0.9
        
        self.mock_entity_recognizer.extract_medical_entities.return_value = {
            'diseases': [mock_entity]
        }
        
        result = self.qa_system.extract_medical_entities("I have diabetes")
        
        self.assertIn('diseases', result)
        self.assertEqual(len(result['diseases']), 1)
        self.assertEqual(result['diseases'][0]['text'], 'diabetes')
        self.assertEqual(result['diseases'][0]['type'], 'disease')
        self.assertEqual(result['diseases'][0]['confidence'], 0.9)
    
    def test_retrieve_relevant_answers_success(self):
        """Test successful answer retrieval."""
        # Mock vector database query
        mock_results = {
            'metadatas': [{'question': 'Test?', 'answer': 'Test answer'}],
            'distances': [0.1],
            'documents': ['Test document'],
            'ids': ['1']
        }
        self.mock_vector_db.query.return_value = mock_results
        
        result = self.qa_system.retrieve_relevant_answers([0.1, 0.2], {}, 5)
        
        self.assertEqual(result, mock_results)
        self.mock_vector_db.query.assert_called_once()
    
    def test_retrieve_relevant_answers_error(self):
        """Test answer retrieval with error."""
        # Mock vector database to raise exception
        self.mock_vector_db.query.side_effect = Exception("DB Error")
        
        result = self.qa_system.retrieve_relevant_answers([0.1, 0.2], {}, 5)
        
        # Should return empty results structure
        self.assertEqual(result['metadatas'], [])
        self.assertEqual(result['distances'], [])
        self.assertEqual(result['documents'], [])
        self.assertEqual(result['ids'], [])
    
    def test_generate_answer_no_docs(self):
        """Test answer generation with no documents."""
        result = self.qa_system.generate_answer("What is diabetes?", {})
        
        self.assertIn("couldn't find relevant information", result)
    
    def test_generate_answer_with_docs(self):
        """Test answer generation with documents."""
        retrieved_docs = {
            'metadatas': [{
                'answer': 'Diabetes is a metabolic disorder.',
                'source': 'Medical Source'
            }]
        }
        
        result = self.qa_system.generate_answer("What is diabetes?", retrieved_docs)
        
        self.assertIn('Diabetes is a metabolic disorder.', result)
        self.assertIn('Medical Source', result)
    
    def test_get_confidence_score(self):
        """Test confidence score calculation."""
        answer = "This is a comprehensive answer about diabetes with detailed information."
        query = "diabetes symptoms treatment"
        
        score = self.qa_system.get_confidence_score(answer, query)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_get_dataset_stats_not_loaded(self):
        """Test getting dataset stats when not loaded."""
        result = self.qa_system.get_dataset_stats()
        
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Dataset not loaded')
    
    def test_get_dataset_stats_loaded(self):
        """Test getting dataset stats when loaded."""
        # Set up system as loaded
        self.qa_system.is_loaded = True
        self.qa_system.qa_pairs = [{'source': 'Source1'}, {'source': 'Source2'}]
        
        # Mock collection stats
        self.mock_vector_db.get_collection_stats.return_value = {'document_count': 2}
        
        result = self.qa_system.get_dataset_stats()
        
        self.assertEqual(result['total_qa_pairs'], 2)
        self.assertEqual(result['vector_db_documents'], 2)
        self.assertIn('sources', result)


class TestMedicalQASystemIntegration(unittest.TestCase):
    """Integration tests for MedicalQASystem."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create system with real dependencies but no actual loading
        self.qa_system = MedicalQASystem(
            enable_sentiment=False,
            enable_multilingual=False
        )
    
    def test_system_initialization_real(self):
        """Test system initialization with real dependencies."""
        self.assertIsNotNone(self.qa_system.vector_db)
        self.assertIsNotNone(self.qa_system.entity_recognizer)
        self.assertIsNotNone(self.qa_system.embedding_service)
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation with real data."""
        # Test with matching terms
        answer = "Diabetes is a chronic disease that affects blood sugar levels."
        query = "diabetes blood sugar"
        score = self.qa_system.get_confidence_score(answer, query)
        
        self.assertGreater(score, 0.5)  # Should have decent confidence
        
        # Test with non-matching terms
        answer = "The weather is sunny today."
        query = "diabetes symptoms"
        score = self.qa_system.get_confidence_score(answer, query)
        
        self.assertLess(score, 0.7)  # Should have lower confidence


if __name__ == '__main__':
    unittest.main()