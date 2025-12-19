"""
Embedding Service
Generates vector embeddings from text using sentence-transformers.
Provides consistent, high-quality embeddings for the vector database.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import numpy as np
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings.
    
    Key Concepts:
    - Embedding: A numerical representation of text
    - Dimension: The size of the embedding vector (e.g., 384, 768)
    - Semantic similarity: Similar meanings have similar embeddings
    
    Why sentence-transformers?
    - Optimized for semantic similarity
    - Fast and efficient
    - Works offline (no API calls needed)
    - Consistent results
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformer model
                       Default: all-MiniLM-L6-v2 (384 dimensions, fast, good quality)
                       Alternatives:
                       - all-mpnet-base-v2 (768 dimensions, better quality, slower)
                       - paraphrase-multilingual (for multiple languages)
            device: Device to run on ('cpu' or 'cuda' for GPU)
        """
        self.model_name = model_name
        self.device = device
        
        logger.info(f"🔄 Loading embedding model: {model_name}")
        
        try:
            # Load the model
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"✅ Embedding model loaded successfully")
            logger.info(f"   Dimension: {self.embedding_dimension}")
            logger.info(f"   Device: {device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        This converts text into a numerical vector that captures its meaning.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding
            
        Example:
            >>> service = EmbeddingService()
            >>> embedding = service.generate_embedding("Hello world")
            >>> len(embedding)
            384
        """
        try:
            # Handle empty text
            if not text or not text.strip():
                logger.warning("⚠️ Empty text provided, returning zero vector")
                return [0.0] * self.embedding_dimension
            
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Convert to list of floats
            embedding_list = embedding.tolist()
            
            logger.debug(f"✅ Generated embedding for text ({len(text)} chars)")
            return embedding_list
            
        except Exception as e:
            logger.error(f"❌ Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_dimension
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Batch processing is much faster than processing one at a time!
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            List of embeddings (one per text)
            
        Example:
            >>> service = EmbeddingService()
            >>> texts = ["Hello", "World", "AI is amazing"]
            >>> embeddings = service.generate_embeddings_batch(texts)
            >>> len(embeddings)
            3
        """
        try:
            if not texts:
                logger.warning("⚠️ Empty text list provided")
                return []
            
            # Filter out empty texts and keep track of indices
            valid_texts = []
            valid_indices = []
            for i, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            if not valid_texts:
                logger.warning("⚠️ All texts are empty")
                return [[0.0] * self.embedding_dimension] * len(texts)
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )
            
            # Create result list with zero vectors for empty texts
            result = []
            valid_idx = 0
            for i in range(len(texts)):
                if i in valid_indices:
                    result.append(embeddings[valid_idx].tolist())
                    valid_idx += 1
                else:
                    result.append([0.0] * self.embedding_dimension)
            
            logger.info(f"✅ Generated {len(result)} embeddings in batch")
            return result
            
        except Exception as e:
            logger.error(f"❌ Batch embedding generation failed: {e}")
            # Return zero vectors as fallback
            return [[0.0] * self.embedding_dimension] * len(texts)
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Similarity ranges from -1 to 1:
        - 1.0: Identical meaning
        - 0.0: Unrelated
        - -1.0: Opposite meaning
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between -1 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"❌ Similarity computation failed: {e}")
            return 0.0
    
    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find the most similar embeddings to a query.
        
        This is useful for testing and debugging.
        
        Args:
            query_embedding: The query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        try:
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                sim = self.compute_similarity(query_embedding, candidate)
                similarities.append((i, sim))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Finding similar embeddings failed: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service.
        
        Returns:
            Embedding dimension (e.g., 384, 768)
        """
        return self.embedding_dimension
    
    def embed_documents_for_db(
        self,
        documents: List[str],
        batch_size: int = 32
    ) -> tuple:
        """
        Prepare documents and embeddings for vector database storage.
        
        This is a convenience method that combines document processing
        and embedding generation.
        
        Args:
            documents: List of document texts
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (documents, embeddings) ready for database
        """
        try:
            logger.info(f"📄 Processing {len(documents)} documents for database")
            
            # Generate embeddings
            embeddings = self.generate_embeddings_batch(
                documents,
                batch_size=batch_size,
                show_progress=True
            )
            
            logger.info(f"✅ Prepared {len(documents)} documents with embeddings")
            return documents, embeddings
            
        except Exception as e:
            logger.error(f"❌ Document preparation failed: {e}")
            raise
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate that an embedding has the correct format.
        
        Args:
            embedding: Embedding to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if it's a list
            if not isinstance(embedding, list):
                return False
            
            # Check dimension
            if len(embedding) != self.embedding_dimension:
                logger.warning(f"⚠️ Invalid dimension: {len(embedding)} != {self.embedding_dimension}")
                return False
            
            # Check if all elements are floats
            if not all(isinstance(x, (int, float)) for x in embedding):
                return False
            
            # Check for NaN or infinity
            if any(np.isnan(x) or np.isinf(x) for x in embedding):
                logger.warning("⚠️ Embedding contains NaN or infinity")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Embedding Service\n")
    
    # Initialize service
    print("1️⃣ Initializing embedding service...")
    service = EmbeddingService()
    print(f"   Embedding dimension: {service.get_embedding_dimension()}\n")
    
    # Test single embedding
    print("2️⃣ Testing single embedding generation...")
    text = "Machine learning is a subset of artificial intelligence"
    embedding = service.generate_embedding(text)
    print(f"   Text: {text}")
    print(f"   Embedding length: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}\n")
    
    # Test batch embeddings
    print("3️⃣ Testing batch embedding generation...")
    texts = [
        "Python is a programming language",
        "Deep learning uses neural networks",
        "Natural language processing is fascinating",
        "Computer vision analyzes images"
    ]
    embeddings = service.generate_embeddings_batch(texts, show_progress=False)
    print(f"   Generated {len(embeddings)} embeddings\n")
    
    # Test similarity
    print("4️⃣ Testing similarity computation...")
    text1 = "I love machine learning"
    text2 = "Machine learning is amazing"
    text3 = "I like pizza"
    
    emb1 = service.generate_embedding(text1)
    emb2 = service.generate_embedding(text2)
    emb3 = service.generate_embedding(text3)
    
    sim_12 = service.compute_similarity(emb1, emb2)
    sim_13 = service.compute_similarity(emb1, emb3)
    
    print(f"   Similarity (ML texts): {sim_12:.4f}")
    print(f"   Similarity (ML vs pizza): {sim_13:.4f}")
    print(f"   ✅ ML texts are more similar!\n")
    
    # Test finding most similar
    print("5️⃣ Testing find most similar...")
    query = "artificial intelligence"
    query_emb = service.generate_embedding(query)
    
    candidates = [
        "machine learning algorithms",
        "cooking recipes",
        "neural networks",
        "travel destinations"
    ]
    candidate_embs = service.generate_embeddings_batch(candidates, show_progress=False)
    
    results = service.find_most_similar(query_emb, candidate_embs, top_k=2)
    print(f"   Query: {query}")
    print(f"   Most similar:")
    for idx, score in results:
        print(f"     - {candidates[idx]} (score: {score:.4f})")
    
    print("\n✅ All tests passed!")
    print("\n💡 Key Takeaway: Similar meanings → Similar embeddings!")
