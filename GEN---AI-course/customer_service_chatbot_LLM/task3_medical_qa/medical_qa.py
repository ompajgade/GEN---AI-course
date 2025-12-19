"""
Medical Q&A System - Task 3
Specialized medical question answering using MedQuAD dataset with entity recognition.
Integrates with vector database for retrieval and LLM for answer generation.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from task3_medical_qa.data_loader import MedQuADDataLoader
from task3_medical_qa.entity_recognizer import MedicalEntityRecognizer
from shared.vector_db_manager import VectorDatabaseManager
from shared.embedding_service import EmbeddingService

# Import sentiment analysis
try:
    sys.path.append(str(Path(__file__).parent.parent / 'task5_sentiment'))
    from sentiment_analysis import SentimentAnalysisEngine
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("Warning: Sentiment analysis not available - continuing without sentiment features")

# Import multilingual support
try:
    sys.path.append(str(Path(__file__).parent.parent / 'task6_multilingual'))
    from multilingual_system import MultiLingualSystem
    MULTILINGUAL_AVAILABLE = True
except ImportError:
    MULTILINGUAL_AVAILABLE = False
    print("Warning: Multilingual support not available - continuing with English only")


class MedicalQASystem:
    """
    Medical Question Answering System using MedQuAD dataset.
    
    Combines entity recognition, vector database retrieval, and answer generation
    to provide accurate medical information.
    """
    
    def __init__(self, 
                 vector_db: Optional[VectorDatabaseManager] = None,
                 entity_recognizer: Optional[MedicalEntityRecognizer] = None,
                 embedding_service: Optional[EmbeddingService] = None,
                 enable_sentiment: bool = True,
                 enable_multilingual: bool = True):
        """
        Initialize the Medical Q&A System.
        
        Args:
            vector_db: Vector database manager instance
            entity_recognizer: Medical entity recognizer instance
            embedding_service: Embedding service instance
            enable_sentiment: Whether to enable sentiment analysis
        """
        self.vector_db = vector_db or VectorDatabaseManager()
        self.entity_recognizer = entity_recognizer or MedicalEntityRecognizer()
        self.embedding_service = embedding_service or EmbeddingService()
        
        # Initialize sentiment analysis if available and enabled
        self.sentiment_engine = None
        if enable_sentiment and SENTIMENT_AVAILABLE:
            try:
                self.sentiment_engine = SentimentAnalysisEngine()
                print("✅ Sentiment analysis enabled for medical Q&A system")
            except Exception as e:
                print(f"Warning: Failed to initialize sentiment analysis: {e}")
        
        # Initialize multilingual support if available and enabled
        self.multilingual_system = None
        if enable_multilingual and MULTILINGUAL_AVAILABLE:
            try:
                self.multilingual_system = MultiLingualSystem()
                print("✅ Multilingual support enabled for medical Q&A system")
            except Exception as e:
                print(f"Warning: Failed to initialize multilingual system: {e}")
        
        self.collection_name = "medical_knowledge"
        self.qa_pairs = []
        self.is_loaded = False
        
        # Check if collection already exists
        self._check_existing_data()
    
    def _check_existing_data(self):
        """Check if the collection already has data loaded"""
        try:
            stats = self.vector_db.get_collection_stats(self.collection_name)
            if stats.get('document_count', 0) > 0:
                self.is_loaded = True
        except:
            self.is_loaded = False
    
    def load_medquad_dataset(self, dataset_path: str = "data/medquad", force_reload: bool = False) -> bool:
        """
        Load and process the MedQuAD dataset into the vector database.
        Checks if data already exists to avoid reloading.
        
        Args:
            dataset_path: Path to the MedQuAD dataset directory
            force_reload: If True, reload even if collection exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if collection already exists and has data
            if not force_reload:
                try:
                    stats = self.vector_db.get_collection_stats(self.collection_name)
                    doc_count = stats.get('document_count', 0)
                    if doc_count > 0:
                        print(f"✅ Dataset already loaded! Found {doc_count} documents in vector database.")
                        print("Skipping reload. Use force_reload=True to reload.")
                        self.is_loaded = True
                        return True
                except:
                    # Collection doesn't exist, proceed with loading
                    pass
            
            print("Loading MedQuAD dataset...")
            
            # Try to use cached data first
            from task3_medical_qa.dataset_cache import DatasetCache
            cache = DatasetCache(cache_dir=f"{dataset_path}/cache")
            
            cached_pairs, _ = cache.load()
            if cached_pairs is not None:
                self.qa_pairs = cached_pairs
                print(f"✅ Loaded {len(self.qa_pairs)} Q&A pairs from cache (fast!)")
            else:
                # Load dataset using data loader
                loader = MedQuADDataLoader(data_dir=dataset_path)
                self.qa_pairs = loader.load_all_qa_pairs()
                
                if not self.qa_pairs:
                    print("No Q&A pairs found in dataset")
                    return False
                
                # Preprocess the pairs
                print("Preprocessing Q&A pairs...")
                self.qa_pairs = loader.preprocess_qa_pairs(self.qa_pairs)
                
                # Cache for next time
                stats = loader.get_dataset_statistics(self.qa_pairs)
                cache.save(self.qa_pairs, stats)
                
                print(f"Loaded {len(self.qa_pairs)} Q&A pairs")
            
            # Create collection in vector database (or get existing)
            print("Creating vector database collection...")
            self.vector_db.create_collection(
                name=self.collection_name,
                metadata={"description": "MedQuAD medical Q&A dataset"}
            )
            
            # Generate embeddings and add to vector database
            print("Generating embeddings and storing in vector database...")
            self._add_to_vector_db()
            
            self.is_loaded = True
            print("Dataset loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def _add_to_vector_db(self, batch_size: int = 100):
        """
        Add Q&A pairs to vector database in batches.
        
        Args:
            batch_size: Number of documents to process in each batch
        """
        total = len(self.qa_pairs)
        
        for i in range(0, total, batch_size):
            batch = self.qa_pairs[i:i + batch_size]
            
            # Prepare documents (combine question and answer for better retrieval)
            documents = []
            metadatas = []
            
            for pair in batch:
                # Combine question and answer for embedding
                combined_text = f"Question: {pair['question']}\nAnswer: {pair['answer']}"
                documents.append(combined_text)
                
                # Store metadata
                metadatas.append({
                    'question': pair['question'],
                    'answer': pair['answer'],
                    'source': pair['source'],
                    'url': pair.get('url', ''),
                    'file': pair.get('file', '')
                })
            
            # Generate embeddings
            embeddings = self.embedding_service.generate_embeddings_batch(documents)
            
            # Add to vector database
            self.vector_db.add_documents(
                collection_name=self.collection_name,
                documents=documents,
                embeddings=embeddings,
                metadata=metadatas
            )
            
            if (i + batch_size) % 500 == 0 or (i + batch_size) >= total:
                print(f"  Processed {min(i + batch_size, total)}/{total} documents...")
    
    def process_medical_query(self, query: str, top_k: int = 5) -> Dict:
        """
        Process a medical query with multilingual and sentiment support.
        
        Args:
            query: Medical question from user
            top_k: Number of relevant answers to retrieve
            
        Returns:
            Dictionary containing query results, entities, answers, and sentiment
        """
        if not self.is_loaded:
            return {
                'error': 'Dataset not loaded. Please load the MedQuAD dataset first.',
                'query': query,
                'entities': {},
                'answers': [],
                'sentiment': None,
                'language': 'en'
            }
        
        # Detect and process language
        detected_language = "en"
        translated_query = query
        
        if self.multilingual_system:
            try:
                lang_result = self.multilingual_system.process_multilingual_query(query)
                detected_language = lang_result["detected_language"]
                translated_query = lang_result.get("translation", query)
            except Exception as e:
                print(f"Warning: Language processing failed: {e}")
        
        # Analyze sentiment of the query (use original query for sentiment)
        sentiment_info = None
        if self.sentiment_engine:
            try:
                sentiment_result = self.sentiment_engine.analyze_sentiment(query)
                sentiment_info = {
                    'label': sentiment_result.label,
                    'score': sentiment_result.score,
                    'raw_scores': sentiment_result.raw_scores
                }
            except Exception as e:
                print(f"Warning: Sentiment analysis failed: {e}")
        
        # Extract medical entities from translated query (for better English processing)
        entities = self.extract_medical_entities(translated_query)
        
        # Generate query embedding using translated query
        query_embedding = self.embedding_service.generate_embedding(translated_query)
        
        # Retrieve relevant answers
        try:
            retrieved_docs = self.retrieve_relevant_answers(query_embedding, entities, top_k)
        except Exception as e:
            print(f"Error retrieving answers: {e}")
            import traceback
            traceback.print_exc()
            return {
                'query': query,
                'original_language': detected_language,
                'translated_query': translated_query if detected_language != "en" else None,
                'entities': entities,
                'answers': [],
                'error': str(e)
            }
        
        # Debug: Check what we got
        if not isinstance(retrieved_docs, dict):
            print(f"WARNING: retrieved_docs is not a dict! Type: {type(retrieved_docs)}")
            print(f"Value: {retrieved_docs}")
            return {
                'query': query,
                'original_language': detected_language,
                'entities': entities,
                'answers': [],
                'error': f'Invalid return type from retrieve_relevant_answers: {type(retrieved_docs)}'
            }
        
        # Format results
        answers = []
        if retrieved_docs and 'metadatas' in retrieved_docs:
            metadatas = retrieved_docs.get('metadatas', [])
            distances = retrieved_docs.get('distances', [])
            
            for i, metadata in enumerate(metadatas):
                if isinstance(metadata, dict):
                    distance = distances[i] if i < len(distances) else 0.0
                    answer_text = metadata.get('answer', '')
                    
                    # Apply sentiment-aware response adjustment if available
                    if self.sentiment_engine and sentiment_info:
                        try:
                            answer_text = self.sentiment_engine.adjust_response_tone(
                                answer_text, sentiment_info['label']
                            )
                        except Exception as e:
                            print(f"Warning: Failed to adjust response tone: {e}")
                    
                    # Generate culturally appropriate response in target language
                    final_answer = answer_text
                    if self.multilingual_system and detected_language != "en":
                        try:
                            final_answer = self.multilingual_system.generate_culturally_appropriate_response(
                                answer_text, detected_language
                            )
                        except Exception as e:
                            print(f"Warning: Cultural adaptation failed: {e}")
                    
                    answers.append({
                        'question': metadata.get('question', ''),
                        'answer': final_answer,
                        'original_answer': metadata.get('answer', ''),
                        'source': metadata.get('source', ''),
                        'url': metadata.get('url', ''),
                        'similarity_score': distance,
                        'confidence': self.get_confidence_score(metadata.get('answer', ''), translated_query)
                    })
        
        return {
            'query': query,
            'original_language': detected_language,
            'translated_query': translated_query if detected_language != "en" else None,
            'entities': entities,
            'answers': answers,
            'num_results': len(answers),
            'sentiment': sentiment_info,
            'sentiment_enabled': self.sentiment_engine is not None,
            'multilingual_enabled': self.multilingual_system is not None
        }
    
    def extract_medical_entities(self, text: str) -> Dict:
        """
        Extract medical entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of extracted entities by type
        """
        entities = self.entity_recognizer.extract_medical_entities(text)
        
        # Convert to serializable format
        result = {}
        for entity_type, entity_list in entities.items():
            result[entity_type] = [
                {
                    'text': e.text,
                    'type': e.entity_type,
                    'confidence': e.confidence
                }
                for e in entity_list
            ]
        
        return result
    
    def retrieve_relevant_answers(self, 
                                  query_embedding: List[float],
                                  entities: Dict,
                                  top_k: int = 5) -> Dict:
        """
        Retrieve relevant answers from vector database.
        
        Args:
            query_embedding: Embedding vector of the query
            entities: Extracted medical entities from query
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with retrieved documents from vector DB
        """
        try:
            # Query vector database
            results = self.vector_db.query(
                collection_name=self.collection_name,
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            # Validate results
            if not isinstance(results, dict):
                print(f"ERROR: Vector DB returned non-dict: {type(results)}")
                return {
                    'metadatas': [],
                    'distances': [],
                    'documents': [],
                    'ids': []
                }
            
            return results
        except Exception as e:
            print(f"Error querying vector database: {e}")
            import traceback
            traceback.print_exc()
            # Return empty results structure
            return {
                'metadatas': [],
                'distances': [],
                'documents': [],
                'ids': []
            }
    
    def generate_answer(self, query: str, retrieved_docs: Dict) -> str:
        """
        Generate a comprehensive answer from retrieved documents.
        
        Args:
            query: Original user query
            retrieved_docs: Dictionary with retrieved documents from vector DB
            
        Returns:
            Generated answer text
        """
        if not retrieved_docs or 'metadatas' not in retrieved_docs or not retrieved_docs['metadatas']:
            return "I couldn't find relevant information to answer your question. Please try rephrasing or ask a different medical question."
        
        # Use the most relevant answer (first result)
        metadatas = retrieved_docs['metadatas']
        answer = metadatas[0].get('answer', '')
        source = metadatas[0].get('source', 'Unknown')
        
        # Format the answer with source attribution
        formatted_answer = f"{answer}\n\n(Source: {source})"
        
        return formatted_answer
    
    def get_confidence_score(self, answer: str, query: str) -> float:
        """
        Calculate confidence score for an answer.
        
        Args:
            answer: Answer text
            query: Original query
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple heuristic based on answer length and query terms
        confidence = 0.5
        
        # Longer answers tend to be more comprehensive
        if len(answer) > 200:
            confidence += 0.2
        elif len(answer) > 500:
            confidence += 0.3
        
        # Check if query terms appear in answer
        query_terms = set(query.lower().split())
        answer_lower = answer.lower()
        
        matching_terms = sum(1 for term in query_terms if term in answer_lower)
        if matching_terms > 0:
            confidence += min(0.2, matching_terms * 0.05)
        
        return min(confidence, 1.0)
    
    def get_dataset_stats(self) -> Dict:
        """
        Get statistics about the loaded dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.is_loaded:
            return {'error': 'Dataset not loaded'}
        
        # Get collection stats from vector database
        collection_stats = self.vector_db.get_collection_stats(self.collection_name)
        doc_count = collection_stats.get('document_count', 0)
        
        # Calculate source distribution from qa_pairs if available
        sources = {}
        if self.qa_pairs:
            for pair in self.qa_pairs:
                source = pair['source']
                sources[source] = sources.get(source, 0) + 1
        
        # If qa_pairs is empty (data was already loaded), use vector DB count
        total_pairs = len(self.qa_pairs) if self.qa_pairs else doc_count
        
        return {
            'total_qa_pairs': total_pairs,
            'vector_db_documents': doc_count,
            'sources': sources,
            'num_sources': len(sources) if sources else 0
        }


if __name__ == "__main__":
    # Example usage
    print("Medical Q&A System Example")
    print("=" * 70)
    
    # Initialize system
    qa_system = MedicalQASystem()
    
    # Load dataset (using a small subset for demo)
    print("\nNote: Loading full dataset takes time. For demo, load manually if needed.")
    
    # Example query processing (without loading full dataset)
    sample_query = "What are the symptoms of diabetes?"
    
    print(f"\nSample Query: {sample_query}")
    print("\nExtracting medical entities...")
    
    entities = qa_system.extract_medical_entities(sample_query)
    
    print("\nExtracted Entities:")
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"  {entity_type}:")
            for entity in entity_list:
                print(f"    - {entity['text']} (confidence: {entity['confidence']:.2f})")
    
    print("\n" + "=" * 70)
    print("Medical Q&A System initialized successfully!")
    print("\nTo use the system:")
    print("  1. Load dataset: qa_system.load_medquad_dataset()")
    print("  2. Process query: qa_system.process_medical_query('your question')")
