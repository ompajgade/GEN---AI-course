"""
Unit tests for Domain Expert System (Task 4)
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from task4_domain_expert.domain_expert import DomainExpertSystem


class TestDomainExpertSystem(unittest.TestCase):
    """Test cases for DomainExpertSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_vector_db = Mock()
        self.mock_llm = Mock()
        self.mock_embedding_service = Mock()
        
        # Initialize system with mocked dependencies
        self.expert_system = DomainExpertSystem(
            domain="computer_science",
            vector_db=self.mock_vector_db,
            llm=self.mock_llm,
            embedding_service=self.mock_embedding_service
        )
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.expert_system)
        self.assertEqual(self.expert_system.domain, "computer_science")
        self.assertEqual(self.expert_system.papers_collection, "computer_science_papers")
        self.assertEqual(self.expert_system.conversations_collection, "computer_science_conversations")
        self.assertIsInstance(self.expert_system.conversation_contexts, dict)
    
    def test_search_papers_success(self):
        """Test successful paper search."""
        # Mock embedding generation
        self.mock_embedding_service.generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock vector database query
        mock_metadata = {
            'paper_id': 'paper_1',
            'title': 'Test Paper',
            'abstract': 'Test abstract',
            'authors': '["Author 1", "Author 2"]',
            'categories': '["cs.AI"]',
            'primary_category': 'cs.AI',
            'category_name': 'Artificial Intelligence',
            'published_date': '2023-01-01',
            'pdf_link': 'http://test.com/paper.pdf'
        }
        
        self.mock_vector_db.query.return_value = {
            'metadatas': [mock_metadata],
            'distances': [0.1]
        }
        
        papers = self.expert_system.search_papers("machine learning", top_k=5)
        
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]['paper_id'], 'paper_1')
        self.assertEqual(papers[0]['title'], 'Test Paper')
        self.assertEqual(papers[0]['similarity_score'], 0.9)  # 1 - 0.1
        self.assertEqual(papers[0]['relevance_rank'], 1)
    
    def test_search_papers_no_results(self):
        """Test paper search with no results."""
        # Mock empty results
        self.mock_vector_db.query.return_value = None
        
        papers = self.expert_system.search_papers("nonexistent topic")
        
        self.assertEqual(len(papers), 0)
    
    def test_search_papers_error(self):
        """Test paper search with error."""
        # Mock vector database to raise exception
        self.mock_vector_db.query.side_effect = Exception("DB Error")
        
        papers = self.expert_system.search_papers("machine learning")
        
        self.assertEqual(len(papers), 0)
    
    def test_summarize_paper_with_data(self):
        """Test paper summarization with paper data."""
        paper_data = {
            'title': 'Test Paper',
            'authors': ['Author 1', 'Author 2'],
            'category_name': 'Artificial Intelligence',
            'primary_category': 'cs.AI',
            'abstract': 'This is a test abstract.'
        }
        
        # Mock LLM response
        self.mock_llm.generate_text.return_value = "This is a test summary."
        
        summary = self.expert_system.summarize_paper(paper_data=paper_data)
        
        self.assertEqual(summary, "This is a test summary.")
        self.mock_llm.generate_text.assert_called_once()
    
    def test_summarize_paper_no_data(self):
        """Test paper summarization without data."""
        summary = self.expert_system.summarize_paper()
        
        self.assertIn("No paper specified", summary)
    
    def test_summarize_paper_error(self):
        """Test paper summarization with LLM error."""
        paper_data = {
            'title': 'Test Paper',
            'authors': ['Author 1'],
            'abstract': 'Test abstract.'
        }
        
        # Mock LLM to raise exception
        self.mock_llm.generate_text.side_effect = Exception("LLM Error")
        
        summary = self.expert_system.summarize_paper(paper_data=paper_data)
        
        self.assertIn("Error generating summary", summary)
    
    def test_explain_concept_success(self):
        """Test successful concept explanation."""
        # Mock paper search
        self.expert_system.search_papers = Mock(return_value=[
            {
                'title': 'Relevant Paper',
                'abstract': 'This paper explains the concept.'
            }
        ])
        
        # Mock LLM response
        self.mock_llm.generate_text.return_value = "This is a concept explanation."
        
        explanation = self.expert_system.explain_concept("neural networks")
        
        self.assertEqual(explanation, "This is a concept explanation.")
        self.mock_llm.generate_text.assert_called_once()
    
    def test_explain_concept_error(self):
        """Test concept explanation with error."""
        # Mock LLM to raise exception
        self.mock_llm.generate_text.side_effect = Exception("LLM Error")
        
        explanation = self.expert_system.explain_concept("neural networks")
        
        self.assertIn("Error generating explanation", explanation)
    
    def test_handle_followup_success(self):
        """Test successful follow-up handling."""
        conversation_id = "test_conv_1"
        
        # Add some context
        self.expert_system.conversation_contexts[conversation_id] = {
            'messages': [
                {'role': 'user', 'content': 'What is AI?'},
                {'role': 'assistant', 'content': 'AI is artificial intelligence.'}
            ]
        }
        
        # Mock paper search
        self.expert_system.search_papers = Mock(return_value=[
            {'title': 'AI Paper'}
        ])
        
        # Mock LLM response
        self.mock_llm.generate_text.return_value = "This is a follow-up response."
        
        response = self.expert_system.handle_followup("Can you elaborate?", conversation_id)
        
        self.assertEqual(response, "This is a follow-up response.")
        self.mock_llm.generate_text.assert_called_once()
    
    def test_handle_followup_error(self):
        """Test follow-up handling with error."""
        # Mock LLM to raise exception
        self.mock_llm.generate_text.side_effect = Exception("LLM Error")
        
        response = self.expert_system.handle_followup("Can you elaborate?", "test_conv")
        
        self.assertIn("Error handling follow-up", response)
    
    def test_update_conversation_context(self):
        """Test conversation context updating."""
        conversation_id = "test_conv_2"
        user_message = "What is machine learning?"
        assistant_response = "Machine learning is a subset of AI."
        
        self.expert_system._update_conversation_context(
            conversation_id, user_message, assistant_response
        )
        
        self.assertIn(conversation_id, self.expert_system.conversation_contexts)
        context = self.expert_system.conversation_contexts[conversation_id]
        self.assertEqual(len(context['messages']), 2)
        self.assertEqual(context['messages'][0]['role'], 'user')
        self.assertEqual(context['messages'][0]['content'], user_message)
        self.assertEqual(context['messages'][1]['role'], 'assistant')
        self.assertEqual(context['messages'][1]['content'], assistant_response)
    
    def test_conversation_context_limit(self):
        """Test conversation context message limit."""
        conversation_id = "test_conv_3"
        
        # Add more than 20 messages
        for i in range(25):
            self.expert_system._update_conversation_context(
                conversation_id, f"User message {i}", f"Assistant response {i}"
            )
        
        context = self.expert_system.conversation_contexts[conversation_id]
        # Should keep only last 20 messages
        self.assertEqual(len(context['messages']), 20)
        # Should start from message 5 (since we added 25 messages, last 20 would be 5-24)
        self.assertIn("User message 15", context['messages'][0]['content'])
    
    def test_extract_key_information_success(self):
        """Test key information extraction."""
        paper_text = "This is a research paper about neural networks."
        
        # Mock LLM response
        self.mock_llm.generate_text.return_value = '{"research_problem": "Neural networks"}'
        
        key_info = self.expert_system.extract_key_information(paper_text)
        
        self.assertIn('research_problem', key_info)
        self.assertEqual(key_info['research_problem'], 'Neural networks')
    
    def test_extract_key_information_invalid_json(self):
        """Test key information extraction with invalid JSON."""
        paper_text = "This is a research paper."
        
        # Mock LLM response with invalid JSON
        self.mock_llm.generate_text.return_value = "This is not JSON"
        
        key_info = self.expert_system.extract_key_information(paper_text)
        
        self.assertIn('extracted_text', key_info)
        self.assertEqual(key_info['extraction_method'], 'text_format')
    
    def test_extract_key_information_error(self):
        """Test key information extraction with error."""
        # Mock LLM to raise exception
        self.mock_llm.generate_text.side_effect = Exception("LLM Error")
        
        key_info = self.expert_system.extract_key_information("Test text")
        
        self.assertIn('error', key_info)
    
    def test_get_domain_statistics_success(self):
        """Test getting domain statistics."""
        # Mock collection stats
        self.mock_vector_db.get_collection_stats.return_value = {'count': 100}
        
        # Add some conversations
        self.expert_system.conversation_contexts['conv1'] = {}
        self.expert_system.conversation_contexts['conv2'] = {}
        
        stats = self.expert_system.get_domain_statistics()
        
        self.assertEqual(stats['domain'], 'computer_science')
        self.assertEqual(stats['papers_loaded'], 100)
        self.assertEqual(stats['active_conversations'], 2)
        self.assertEqual(stats['system_status'], 'active')
    
    def test_get_domain_statistics_error(self):
        """Test getting domain statistics with error."""
        # Mock vector database to raise exception
        self.mock_vector_db.get_collection_stats.side_effect = Exception("DB Error")
        
        stats = self.expert_system.get_domain_statistics()
        
        self.assertIn('error', stats)
    
    def test_domain_prompts_exist(self):
        """Test that domain prompts are properly configured."""
        self.assertIn('computer_science', self.expert_system.domain_prompts)
        cs_prompts = self.expert_system.domain_prompts['computer_science']
        
        self.assertIn('summarize', cs_prompts)
        self.assertIn('explain', cs_prompts)
        self.assertIn('followup', cs_prompts)
        
        # Check that prompts are not empty
        self.assertTrue(len(cs_prompts['summarize']) > 0)
        self.assertTrue(len(cs_prompts['explain']) > 0)
        self.assertTrue(len(cs_prompts['followup']) > 0)


class TestDomainExpertSystemIntegration(unittest.TestCase):
    """Integration tests for DomainExpertSystem."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create system with real dependencies but no actual loading
        self.expert_system = DomainExpertSystem(domain="computer_science")
    
    def test_system_initialization_real(self):
        """Test system initialization with real dependencies."""
        self.assertIsNotNone(self.expert_system.vector_db)
        self.assertIsNotNone(self.expert_system.llm)
        self.assertIsNotNone(self.expert_system.embedding_service)
        self.assertIsNotNone(self.expert_system.data_loader)
        self.assertIsNotNone(self.expert_system.dataset_cache)


if __name__ == '__main__':
    unittest.main()