"""
Vector Database Manager
Manages ChromaDB collections for storing and retrieving vector embeddings.
Used by all 6 tasks for efficient similarity search.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDatabaseManager:
    """
    Manages vector database operations using ChromaDB.
    
    Key Concepts:
    - Collection: A group of documents (like a table in SQL)
    - Embedding: Numerical representation of text/images
    - Similarity Search: Find documents similar to a query
    """
    
    def __init__(self, db_type: str = "chromadb", persist_directory: str = "./data/vector_db"):
        """
        Initialize the vector database manager.
        
        Args:
            db_type: Type of vector database (currently only chromadb supported)
            persist_directory: Where to store the database files
        """
        self.db_type = db_type
        self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"✅ Vector database initialized at {persist_directory}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector database: {e}")
            raise
    
    def create_collection(self, name: str, metadata: Optional[Dict] = None) -> Any:
        """
        Create a new collection in the vector database.
        
        A collection is like a table - it groups related documents together.
        Examples: 'medical_knowledge', 'scientific_papers', 'conversation_history'
        
        Args:
            name: Name of the collection
            metadata: Optional metadata about the collection
            
        Returns:
            Collection object
        """
        try:
            # Check if collection already exists
            existing_collections = [col.name for col in self.client.list_collections()]
            
            if name in existing_collections:
                logger.info(f"📂 Collection '{name}' already exists, retrieving it")
                return self.client.get_collection(name=name)
            
            # Create new collection
            collection = self.client.create_collection(
                name=name,
                metadata=metadata or {"created_at": datetime.now().isoformat()}
            )
            logger.info(f"✅ Created new collection: {name}")
            return collection
            
        except Exception as e:
            logger.error(f"❌ Failed to create collection '{name}': {e}")
            raise
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents with their embeddings to a collection.
        
        Args:
            collection_name: Name of the collection
            documents: List of text documents
            embeddings: List of embedding vectors (one per document)
            metadata: Optional metadata for each document
            ids: Optional custom IDs (auto-generated if not provided)
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # Generate IDs if not provided
            if ids is None:
                existing_count = collection.count()
                ids = [f"{collection_name}_{existing_count + i}" for i in range(len(documents))]
            
            # Add documents to collection
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadata or [{} for _ in documents],
                ids=ids
            )
            
            logger.info(f"✅ Added {len(documents)} documents to '{collection_name}'")
            
        except Exception as e:
            logger.error(f"❌ Failed to add documents to '{collection_name}': {e}")
            raise
    
    def query(
        self,
        collection_name: str,
        query_embedding: List[float] = None,
        query_embeddings: List[List[float]] = None,
        top_k: int = 5,
        n_results: int = None,
        where: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Query the collection for similar documents.
        
        This is the core similarity search function!
        
        Args:
            collection_name: Name of the collection to search
            query_embedding: Single embedding vector of the query (deprecated, use query_embeddings)
            query_embeddings: List of embedding vectors for batch queries
            top_k: Number of results to return
            n_results: Alternative parameter name for top_k (for compatibility)
            where: Optional filter conditions
            
        Returns:
            Dictionary with 'documents', 'metadatas', 'distances', 'ids'
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # Handle both parameter formats
            if query_embeddings is None and query_embedding is not None:
                query_embeddings = [query_embedding]
            elif query_embeddings is None:
                raise ValueError("Either query_embedding or query_embeddings must be provided")
            
            # Use n_results if provided, otherwise use top_k
            num_results = n_results if n_results is not None else top_k
            
            # Perform similarity search
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=num_results,
                where=where
            )
            
            logger.info(f"🔍 Found {len(results['documents'][0])} results in '{collection_name}'")
            
            return {
                'documents': results['documents'][0],
                'metadatas': results['metadatas'][0],
                'distances': results['distances'][0],
                'ids': results['ids'][0]
            }
            
        except Exception as e:
            logger.error(f"❌ Query failed for '{collection_name}': {e}")
            raise
    
    def update_collection(
        self,
        collection_name: str,
        new_documents: List[str],
        new_embeddings: List[List[float]],
        new_metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Update a collection with new documents.
        
        This is used by Task 1 (Dynamic Knowledge Base Expansion)
        to add new information over time.
        
        Args:
            collection_name: Name of the collection
            new_documents: New documents to add
            new_embeddings: Embeddings for new documents
            new_metadata: Optional metadata for new documents
        """
        try:
            # Simply add new documents (ChromaDB handles duplicates)
            self.add_documents(
                collection_name=collection_name,
                documents=new_documents,
                embeddings=new_embeddings,
                metadata=new_metadata
            )
            logger.info(f"✅ Updated collection '{collection_name}' with {len(new_documents)} new documents")
            
        except Exception as e:
            logger.error(f"❌ Failed to update collection '{collection_name}': {e}")
            raise
    
    def delete_collection(self, name: str) -> None:
        """
        Delete a collection from the database.
        
        Args:
            name: Name of the collection to delete
        """
        try:
            self.client.delete_collection(name=name)
            logger.info(f"🗑️ Deleted collection: {name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to delete collection '{name}': {e}")
            raise
    
    def get_collection_stats(self, name: str) -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Args:
            name: Name of the collection
            
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.client.get_collection(name=name)
            count = collection.count()
            metadata = collection.metadata
            
            stats = {
                'name': name,
                'document_count': count,
                'metadata': metadata
            }
            
            logger.info(f"📊 Collection '{name}' has {count} documents")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get stats for '{name}': {e}")
            raise
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the database.
        
        Returns:
            List of collection names
        """
        try:
            collections = [col.name for col in self.client.list_collections()]
            logger.info(f"📋 Found {len(collections)} collections: {collections}")
            return collections
            
        except Exception as e:
            logger.error(f"❌ Failed to list collections: {e}")
            raise
    
    def reset_database(self) -> None:
        """
        Reset the entire database (delete all collections).
        
        ⚠️ WARNING: This deletes all data!
        """
        try:
            self.client.reset()
            logger.warning("⚠️ Database reset - all collections deleted!")
            
        except Exception as e:
            logger.error(f"❌ Failed to reset database: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Vector Database Manager\n")
    
    # Initialize manager
    db_manager = VectorDatabaseManager()
    
    # Create a test collection
    print("1️⃣ Creating test collection...")
    collection = db_manager.create_collection(
        name="test_collection",
        metadata={"purpose": "testing"}
    )
    
    # Add some test documents
    print("\n2️⃣ Adding test documents...")
    test_docs = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language"
    ]
    # Note: In real usage, these would be actual embeddings from a model
    # For testing, we use dummy embeddings
    test_embeddings = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7]
    ]
    
    db_manager.add_documents(
        collection_name="test_collection",
        documents=test_docs,
        embeddings=test_embeddings
    )
    
    # Get collection stats
    print("\n3️⃣ Getting collection stats...")
    stats = db_manager.get_collection_stats("test_collection")
    print(f"   Stats: {stats}")
    
    # Query the collection
    print("\n4️⃣ Querying collection...")
    query_embedding = [0.15, 0.25, 0.35, 0.45, 0.55]
    results = db_manager.query(
        collection_name="test_collection",
        query_embedding=query_embedding,
        top_k=2
    )
    print(f"   Found {len(results['documents'])} results")
    for i, doc in enumerate(results['documents']):
        print(f"   Result {i+1}: {doc[:50]}...")
    
    # List all collections
    print("\n5️⃣ Listing all collections...")
    collections = db_manager.list_collections()
    print(f"   Collections: {collections}")
    
    print("\n✅ All tests passed!")
