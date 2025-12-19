"""
Domain Expert System - Task 4
Provides expert-level explanations and paper search for scientific domains.
Uses arXiv dataset for research paper retrieval and summarization.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add shared directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.vector_db_manager import VectorDatabaseManager
from shared.llm_integration import LLMIntegration
from shared.embedding_service import EmbeddingService
from shared.utils import load_json, save_json
from task4_domain_expert.data_loader import ArxivDataLoader
from task4_domain_expert.dataset_cache import DatasetCache

# Import sentiment analysis
try:
    sys.path.append(str(Path(__file__).parent.parent / 'task5_sentiment'))
    from sentiment_analysis import SentimentAnalysisEngine
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# Import multilingual support
try:
    sys.path.append(str(Path(__file__).parent.parent / 'task6_multilingual'))
    from multilingual_system import MultiLingualSystem
    MULTILINGUAL_AVAILABLE = True
except ImportError:
    MULTILINGUAL_AVAILABLE = False


class DomainExpertSystem:
    """
    Domain Expert System for scientific paper analysis and explanation.
    
    Provides capabilities for:
    - Paper search and retrieval
    - Paper summarization
    - Concept explanation
    - Follow-up question handling
    - Conversation context management
    """
    
    def __init__(self, 
                 domain: str = "computer_science",
                 vector_db: VectorDatabaseManager = None,
                 llm: LLMIntegration = None,
                 embedding_service: EmbeddingService = None):
        """
        Initialize the Domain Expert System.
        
        Args:
            domain: Scientific domain to focus on
            vector_db: Vector database manager instance
            llm: LLM integration instance
            embedding_service: Embedding service instance
        """
        self.domain = domain
        self.vector_db = vector_db or VectorDatabaseManager()
        self.llm = llm or LLMIntegration()
        self.embedding_service = embedding_service or EmbeddingService()
        self.data_loader = ArxivDataLoader()
        self.dataset_cache = DatasetCache()
        
        # Collection names
        self.papers_collection = f"{domain}_papers"
        self.conversations_collection = f"{domain}_conversations"
        
        # Conversation context
        self.conversation_contexts = {}
        
        # Domain-specific prompts
        self.domain_prompts = {
            "computer_science": {
                "summarize": """You are an expert computer scientist. Summarize this research paper in a clear, accessible way.
                Focus on: methodology, key contributions, results, and implications.
                Make it understandable for both experts and students.""",
                
                "explain": """You are an expert computer scientist. Explain this concept clearly and thoroughly.
                Use examples, analogies, and break down complex ideas into understandable parts.
                Consider the context and provide practical insights.""",
                
                "followup": """You are an expert computer scientist continuing a conversation.
                Use the conversation history to provide contextually relevant answers.
                Build upon previous explanations and maintain consistency."""
            }
        }
        
        print(f"🤖 Domain Expert System initialized for: {domain}")
    
    def load_arxiv_dataset(self, force_reload: bool = False) -> bool:
        """
        Load arXiv dataset into vector database with caching.
        
        Args:
            force_reload: Whether to force reload even if collection exists
            
        Returns:
            True if successful, False otherwise
        """
        # Check if vector database is already loaded (using cache)
        if not force_reload:
            try:
                stats = self.vector_db.get_collection_stats(self.papers_collection)
                if stats and stats.get('document_count', 0) >= 800:  # Expect at least 800 papers for full load
                    print(f"✅ Papers collection already loaded with {stats['document_count']} documents (cached)")
                    print("⚡ Skipping all processing - using existing vector database")
                    
                    # Mark as loaded for future runs if not already marked
                    if not self.dataset_cache.is_vector_db_loaded(self.papers_collection):
                        self.dataset_cache.mark_vector_db_loaded(self.papers_collection, stats['document_count'])
                    
                    return True
            except Exception as e:
                print(f"⚠️ Vector database not accessible: {str(e)}")
                pass  # Collection doesn't exist, continue with loading
        
        print("📥 Loading arXiv dataset into vector database...")
        
        # Try to load from cache first
        source_file = "data/arxiv/processed_papers.json"
        papers = None
        
        if not force_reload and self.dataset_cache.is_papers_cache_valid(source_file):
            papers = self.dataset_cache.load_papers_cache()
        
        # If cache miss, load from source
        if papers is None:
            print("📄 Loading papers from source file...")
            papers = self.data_loader.load_processed_papers()
            if not papers:
                print("❌ No processed papers found. Please run data_loader.py first.")
                return False
            
            # Save to cache for next time
            self.dataset_cache.save_papers_cache(papers, source_file)
        else:
            print("⚡ Using cached papers data")
        
        print(f"📊 Processing {len(papers)} papers for embedding...")
        
        # Create collection
        collection_metadata = {
            "description": f"Scientific papers for {self.domain} domain expert system",
            "domain": self.domain,
            "created_at": datetime.now().isoformat(),
            "paper_count": len(papers)
        }
        
        try:
            self.vector_db.create_collection(self.papers_collection, collection_metadata)
        except Exception as e:
            print(f"⚠️ Collection might already exist: {str(e)}")
        
        # Process papers in batches
        batch_size = 50
        successful_embeddings = 0
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            
            try:
                # Prepare documents and metadata
                documents = []
                metadatas = []
                
                for paper in batch:
                    # Use title + abstract for embedding
                    text_for_embedding = paper.get('text_for_embedding', '')
                    if not text_for_embedding:
                        text_for_embedding = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                    
                    documents.append(text_for_embedding)
                    
                    # Prepare metadata
                    metadata = {
                        'paper_id': paper.get('id', ''),
                        'title': paper.get('title', ''),
                        'abstract': paper.get('abstract', ''),
                        'authors': json.dumps(paper.get('authors', [])),
                        'categories': json.dumps(paper.get('categories', [])),
                        'primary_category': paper.get('primary_category', ''),
                        'category_name': paper.get('category_name', ''),
                        'published_date': paper.get('published_date', ''),
                        'pdf_link': paper.get('pdf_link', ''),
                        'processed_date': paper.get('processed_date', '')
                    }
                    metadatas.append(metadata)
                
                # Generate embeddings
                embeddings = []
                for doc in documents:
                    try:
                        embedding = self.embedding_service.generate_embedding(doc)
                        embeddings.append(embedding)
                    except Exception as e:
                        print(f"⚠️ Error generating embedding: {str(e)}")
                        # Use zero vector as fallback
                        embeddings.append([0.0] * 384)  # Default dimension
                
                # Add to vector database
                self.vector_db.add_documents(
                    collection_name=self.papers_collection,
                    documents=documents,
                    embeddings=embeddings,
                    metadata=metadatas
                )
                
                successful_embeddings += len(batch)
                print(f"✅ Processed batch {i//batch_size + 1}/{(len(papers) + batch_size - 1)//batch_size} "
                      f"({successful_embeddings}/{len(papers)} papers)")
                
            except Exception as e:
                print(f"❌ Error processing batch {i//batch_size + 1}: {str(e)}")
                continue
        
        if successful_embeddings > 0:
            print(f"\n✅ Successfully loaded {successful_embeddings} papers into vector database")
            
            # Mark vector database as loaded in cache
            self.dataset_cache.mark_vector_db_loaded(self.papers_collection, successful_embeddings)
            
            return True
        else:
            print("❌ Failed to load any papers")
            return False
    
    def search_papers(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search for relevant papers based on query.
        
        Args:
            query: Search query
            top_k: Number of papers to return
            
        Returns:
            List of relevant papers with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Search vector database
            results = self.vector_db.query(
                collection_name=self.papers_collection,
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            if not results or 'metadatas' not in results:
                print("❌ No results found")
                return []
            
            # Process results
            papers = []
            metadatas = results['metadatas']
            distances = results.get('distances', [])
            
            for i, metadata in enumerate(metadatas):
                distance = distances[i] if i < len(distances) else 0.0
                similarity = 1 - distance  # Convert distance to similarity
                
                paper = {
                    'paper_id': metadata.get('paper_id', ''),
                    'title': metadata.get('title', ''),
                    'abstract': metadata.get('abstract', ''),
                    'authors': json.loads(metadata.get('authors', '[]')),
                    'categories': json.loads(metadata.get('categories', '[]')),
                    'primary_category': metadata.get('primary_category', ''),
                    'category_name': metadata.get('category_name', ''),
                    'published_date': metadata.get('published_date', ''),
                    'pdf_link': metadata.get('pdf_link', ''),
                    'similarity_score': similarity,
                    'relevance_rank': i + 1
                }
                papers.append(paper)
            
            print(f"🔍 Found {len(papers)} relevant papers for query: '{query[:50]}...'")
            return papers
            
        except Exception as e:
            print(f"❌ Error searching papers: {str(e)}")
            return []
    
    def summarize_paper(self, paper_id: str = None, paper_data: Dict = None) -> str:
        """
        Generate a summary of a scientific paper.
        
        Args:
            paper_id: ID of paper to summarize
            paper_data: Paper data dictionary (alternative to paper_id)
            
        Returns:
            Generated summary
        """
        if paper_data is None and paper_id is None:
            return "❌ No paper specified for summarization"
        
        # Get paper data if not provided
        if paper_data is None:
            # Search for paper by ID
            papers = self.search_papers(paper_id, top_k=1)
            if not papers:
                return f"❌ Paper with ID '{paper_id}' not found"
            paper_data = papers[0]
        
        try:
            # Prepare summarization prompt
            domain_prompt = self.domain_prompts.get(self.domain, {}).get("summarize", "Summarize this research paper:")
            
            paper_text = f"""
Title: {paper_data.get('title', '')}

Authors: {', '.join(paper_data.get('authors', []))}

Category: {paper_data.get('category_name', '')} ({paper_data.get('primary_category', '')})

Abstract: {paper_data.get('abstract', '')}
"""
            
            prompt = f"""{domain_prompt}

{paper_text}

Please provide a comprehensive summary that includes:
1. Main research question and motivation
2. Methodology and approach
3. Key findings and results
4. Significance and implications
5. Limitations and future work

Make the summary accessible to both experts and students in the field."""
            
            # Generate summary using LLM
            summary = self.llm.generate_text(prompt, max_tokens=1024)
            
            if summary:
                print(f"📄 Generated summary for: {paper_data.get('title', 'Unknown')[:50]}...")
                return summary
            else:
                return "❌ Failed to generate summary"
                
        except Exception as e:
            print(f"❌ Error generating summary: {str(e)}")
            return f"❌ Error generating summary: {str(e)}"
    
    def explain_concept(self, concept: str, context: str = "") -> str:
        """
        Provide detailed explanation of a scientific concept.
        
        Args:
            concept: Concept to explain
            context: Additional context for the explanation
            
        Returns:
            Generated explanation
        """
        try:
            # Search for relevant papers about the concept
            relevant_papers = self.search_papers(concept, top_k=3)
            
            # Prepare context from papers
            paper_context = ""
            if relevant_papers:
                paper_context = "\n\nRelevant research papers:\n"
                for i, paper in enumerate(relevant_papers[:3], 1):
                    paper_context += f"{i}. {paper['title']}\n"
                    paper_context += f"   Abstract: {paper['abstract'][:200]}...\n\n"
            
            # Prepare explanation prompt
            domain_prompt = self.domain_prompts.get(self.domain, {}).get("explain", "Explain this concept:")
            
            prompt = f"""{domain_prompt}

Concept to explain: {concept}

Additional context: {context}

{paper_context}

Please provide a comprehensive explanation that includes:
1. Clear definition and core principles
2. How it works (methodology/mechanism)
3. Real-world applications and examples
4. Relationship to other concepts
5. Current research trends and developments
6. Practical implications

Use examples and analogies to make complex ideas accessible."""
            
            # Generate explanation using LLM
            explanation = self.llm.generate_text(prompt, max_tokens=1024)
            
            if explanation:
                print(f"💡 Generated explanation for concept: {concept}")
                return explanation
            else:
                return "❌ Failed to generate explanation"
                
        except Exception as e:
            print(f"❌ Error generating explanation: {str(e)}")
            return f"❌ Error generating explanation: {str(e)}"
    
    def handle_followup(self, question: str, conversation_id: str) -> str:
        """
        Handle follow-up questions with conversation context.
        
        Args:
            question: Follow-up question
            conversation_id: ID of the conversation
            
        Returns:
            Generated response
        """
        try:
            # Get conversation context
            context = self.conversation_contexts.get(conversation_id, {})
            
            # Prepare context string
            context_str = ""
            if context.get('messages'):
                context_str = "\n\nConversation history:\n"
                for msg in context['messages'][-5:]:  # Last 5 messages
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:200]  # Truncate long messages
                    context_str += f"{role.capitalize()}: {content}...\n"
            
            # Search for relevant papers if needed
            relevant_papers = self.search_papers(question, top_k=2)
            paper_context = ""
            if relevant_papers:
                paper_context = "\n\nRelevant papers:\n"
                for paper in relevant_papers[:2]:
                    paper_context += f"- {paper['title']}\n"
            
            # Prepare follow-up prompt
            domain_prompt = self.domain_prompts.get(self.domain, {}).get("followup", "Answer this follow-up question:")
            
            prompt = f"""{domain_prompt}

Current question: {question}

{context_str}

{paper_context}

Please provide a contextually relevant answer that:
1. Builds upon the previous conversation
2. Addresses the specific question asked
3. Maintains consistency with earlier explanations
4. Provides additional insights or clarifications
5. References relevant research when appropriate"""
            
            # Generate response using LLM
            response = self.llm.generate_text(prompt, max_tokens=1024)
            
            if response:
                # Update conversation context
                self._update_conversation_context(conversation_id, question, response)
                print(f"💬 Generated follow-up response for conversation: {conversation_id}")
                return response
            else:
                return "❌ Failed to generate follow-up response"
                
        except Exception as e:
            print(f"❌ Error handling follow-up: {str(e)}")
            return f"❌ Error handling follow-up: {str(e)}"
    
    def _update_conversation_context(self, conversation_id: str, user_message: str, assistant_response: str):
        """
        Update conversation context with new messages.
        
        Args:
            conversation_id: ID of the conversation
            user_message: User's message
            assistant_response: Assistant's response
        """
        if conversation_id not in self.conversation_contexts:
            self.conversation_contexts[conversation_id] = {
                'created_at': datetime.now().isoformat(),
                'messages': []
            }
        
        context = self.conversation_contexts[conversation_id]
        
        # Add user message
        context['messages'].append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Add assistant response
        context['messages'].append({
            'role': 'assistant',
            'content': assistant_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 20 messages to manage memory
        if len(context['messages']) > 20:
            context['messages'] = context['messages'][-20:]
        
        context['updated_at'] = datetime.now().isoformat()
    
    def extract_key_information(self, paper_text: str) -> Dict:
        """
        Extract key information from paper text.
        
        Args:
            paper_text: Full text of the paper
            
        Returns:
            Dictionary with extracted information
        """
        try:
            prompt = f"""Extract key information from this research paper:

{paper_text[:2000]}...

Please extract and structure the following information:
1. Research problem/question
2. Methodology/approach
3. Key contributions
4. Main results/findings
5. Limitations
6. Future work suggestions

Format as JSON with clear categories."""
            
            response = self.llm.generate_text(prompt, max_tokens=512)
            
            # Try to parse as JSON, fallback to structured text
            try:
                import json
                key_info = json.loads(response)
            except:
                # Fallback to structured text
                key_info = {
                    'extracted_text': response,
                    'extraction_method': 'text_format'
                }
            
            return key_info
            
        except Exception as e:
            print(f"❌ Error extracting key information: {str(e)}")
            return {'error': str(e)}
    
    def get_domain_statistics(self) -> Dict:
        """
        Get statistics about the domain expert system.
        
        Returns:
            Dictionary with system statistics
        """
        try:
            # Get collection stats
            collection_stats = self.vector_db.get_collection_stats(self.papers_collection)
            
            # Get conversation stats
            conversation_count = len(self.conversation_contexts)
            
            stats = {
                'domain': self.domain,
                'papers_loaded': collection_stats.get('count', 0) if collection_stats else 0,
                'active_conversations': conversation_count,
                'collection_name': self.papers_collection,
                'system_status': 'active' if collection_stats and collection_stats.get('count', 0) > 0 else 'inactive'
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting statistics: {str(e)}")
            return {'error': str(e)}


def main():
    """
    Main function to demonstrate the Domain Expert System.
    """
    print("🤖 Domain Expert System - Task 4")
    print("=" * 50)
    
    # Initialize system
    expert_system = DomainExpertSystem(domain="computer_science")
    
    # Load dataset
    print("\n📥 Loading arXiv dataset...")
    if expert_system.load_arxiv_dataset():
        print("✅ Dataset loaded successfully")
        
        # Get system statistics
        stats = expert_system.get_domain_statistics()
        print(f"\n📊 System Statistics:")
        print(f"   Domain: {stats['domain']}")
        print(f"   Papers loaded: {stats['papers_loaded']}")
        print(f"   Status: {stats['system_status']}")
        
        # Test paper search
        print("\n🔍 Testing paper search...")
        query = "machine learning neural networks"
        papers = expert_system.search_papers(query, top_k=3)
        
        if papers:
            print(f"\nTop 3 papers for '{query}':")
            for i, paper in enumerate(papers[:3], 1):
                print(f"\n{i}. {paper['title']}")
                print(f"   Category: {paper['category_name']}")
                print(f"   Similarity: {paper['similarity_score']:.3f}")
                print(f"   Abstract: {paper['abstract'][:150]}...")
            
            # Test summarization
            print(f"\n📄 Testing paper summarization...")
            summary = expert_system.summarize_paper(paper_data=papers[0])
            print(f"\nSummary of '{papers[0]['title'][:50]}...':")
            print(summary[:300] + "..." if len(summary) > 300 else summary)
        
        # Test concept explanation
        print(f"\n💡 Testing concept explanation...")
        concept = "transformer architecture"
        explanation = expert_system.explain_concept(concept)
        print(f"\nExplanation of '{concept}':")
        print(explanation[:300] + "..." if len(explanation) > 300 else explanation)
        
    else:
        print("❌ Failed to load dataset")


if __name__ == "__main__":
    main()