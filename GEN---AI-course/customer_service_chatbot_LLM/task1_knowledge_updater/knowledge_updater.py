"""
Knowledge Base Updater - Task 1
Dynamically expands the chatbot's knowledge base by fetching and processing
new information from various sources.

This is the core of Task 1: Dynamic Knowledge Base Expansion
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.vector_db_manager import VectorDatabaseManager
from shared.embedding_service import EmbeddingService
from shared.utils import (
    save_json, load_json, get_timestamp, 
    ProgressTracker, clean_text, generate_id
)
import requests
from bs4 import BeautifulSoup
import feedparser
import schedule
import threading
from typing import List, Dict, Any, Optional, Callable
import logging
from datetime import datetime
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeBaseUpdater:
    """
    Dynamically updates the chatbot's knowledge base.
    
    Key Features:
    - Fetch content from multiple sources (RSS, web, files)
    - Process and clean text
    - Generate embeddings
    - Update vector database
    - Track update history
    
    This enables the chatbot to stay current with new information!
    """
    
    def __init__(
        self,
        vector_db: VectorDatabaseManager,
        embedding_service: EmbeddingService,
        collection_name: str = "general_knowledge"
    ):
        """
        Initialize the Knowledge Base Updater.
        
        Args:
            vector_db: Vector database manager instance
            embedding_service: Embedding service instance
            collection_name: Name of the collection to update
        """
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.collection_name = collection_name
        
        # Create collection if it doesn't exist
        self.vector_db.create_collection(
            name=collection_name,
            metadata={"purpose": "dynamic_knowledge_base"}
        )
        
        # Track sources and update history
        self.sources = {}
        self.update_history = []
        
        logger.info(f"✅ Knowledge Base Updater initialized")
        logger.info(f"   Collection: {collection_name}")
    
    def add_source(
        self,
        source_id: str,
        source_type: str,
        source_config: Dict[str, Any]
    ) -> None:
        """
        Add a new data source to monitor.
        
        Source Types:
        - 'rss': RSS feed URL
        - 'web': Web page URL to scrape
        - 'file': Local file path
        - 'api': API endpoint
        
        Args:
            source_id: Unique identifier for the source
            source_type: Type of source (rss, web, file, api)
            source_config: Configuration dictionary for the source
        """
        self.sources[source_id] = {
            'type': source_type,
            'config': source_config,
            'added_at': get_timestamp(),
            'last_updated': None,
            'total_documents': 0
        }
        
        logger.info(f"✅ Added source: {source_id} (type: {source_type})")
    
    def fetch_new_content(self, source_id: str) -> List[Dict[str, Any]]:
        """
        Fetch new content from a specific source.
        
        Args:
            source_id: ID of the source to fetch from
            
        Returns:
            List of documents with 'text' and 'metadata'
        """
        if source_id not in self.sources:
            logger.error(f"❌ Source not found: {source_id}")
            return []
        
        source = self.sources[source_id]
        source_type = source['type']
        config = source['config']
        
        logger.info(f"🔄 Fetching content from: {source_id}")
        
        try:
            if source_type == 'rss':
                return self._fetch_from_rss(config)
            elif source_type == 'web':
                return self._fetch_from_web(config)
            elif source_type == 'file':
                return self._fetch_from_file(config)
            elif source_type == 'api':
                return self._fetch_from_api(config)
            else:
                logger.error(f"❌ Unknown source type: {source_type}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch from {source_id}: {e}")
            return []
    
    def _fetch_from_rss(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch content from RSS feed.
        
        Args:
            config: Dictionary with 'url' key
            
        Returns:
            List of documents
        """
        url = config.get('url')
        if not url:
            logger.error("❌ RSS config missing 'url'")
            return []
        
        try:
            feed = feedparser.parse(url)
            documents = []
            
            for entry in feed.entries[:config.get('max_items', 10)]:
                doc = {
                    'text': f"{entry.title}\n\n{entry.get('summary', '')}",
                    'metadata': {
                        'title': entry.title,
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source_type': 'rss'
                    }
                }
                documents.append(doc)
            
            logger.info(f"✅ Fetched {len(documents)} items from RSS feed")
            return documents
            
        except Exception as e:
            logger.error(f"❌ RSS fetch failed: {e}")
            return []
    
    def _fetch_from_web(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch content from web page.
        
        Args:
            config: Dictionary with 'url' and optional 'selector'
            
        Returns:
            List of documents
        """
        url = config.get('url')
        if not url:
            logger.error("❌ Web config missing 'url'")
            return []
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text from specific selector or entire page
            selector = config.get('selector')
            if selector:
                elements = soup.select(selector)
                texts = [elem.get_text(strip=True) for elem in elements]
            else:
                # Get all paragraph text
                texts = [p.get_text(strip=True) for p in soup.find_all('p')]
            
            # Combine into documents
            documents = []
            for i, text in enumerate(texts):
                if text and len(text) > 50:  # Filter out very short text
                    doc = {
                        'text': clean_text(text),
                        'metadata': {
                            'url': url,
                            'index': i,
                            'source_type': 'web'
                        }
                    }
                    documents.append(doc)
            
            logger.info(f"✅ Fetched {len(documents)} items from web page")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Web fetch failed: {e}")
            return []
    
    def _fetch_from_file(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch content from local file.
        
        Args:
            config: Dictionary with 'path' key
            
        Returns:
            List of documents
        """
        file_path = config.get('path')
        if not file_path or not os.path.exists(file_path):
            logger.error(f"❌ File not found: {file_path}")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks if needed
            chunk_size = config.get('chunk_size', 1000)
            chunks = [content[i:i+chunk_size] 
                     for i in range(0, len(content), chunk_size)]
            
            documents = []
            for i, chunk in enumerate(chunks):
                doc = {
                    'text': clean_text(chunk),
                    'metadata': {
                        'file_path': file_path,
                        'chunk_index': i,
                        'source_type': 'file'
                    }
                }
                documents.append(doc)
            
            logger.info(f"✅ Fetched {len(documents)} chunks from file")
            return documents
            
        except Exception as e:
            logger.error(f"❌ File fetch failed: {e}")
            return []
    
    def _fetch_from_api(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch content from API endpoint.
        
        Args:
            config: Dictionary with 'url' and optional 'headers', 'params'
            
        Returns:
            List of documents
        """
        url = config.get('url')
        if not url:
            logger.error("❌ API config missing 'url'")
            return []
        
        try:
            response = requests.get(
                url,
                headers=config.get('headers', {}),
                params=config.get('params', {}),
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract documents based on config
            text_field = config.get('text_field', 'text')
            documents = []
            
            # Handle list of items
            if isinstance(data, list):
                items = data
            else:
                items = data.get(config.get('items_key', 'items'), [])
            
            for item in items:
                if isinstance(item, dict) and text_field in item:
                    doc = {
                        'text': clean_text(str(item[text_field])),
                        'metadata': {
                            'source_type': 'api',
                            **{k: v for k, v in item.items() if k != text_field}
                        }
                    }
                    documents.append(doc)
            
            logger.info(f"✅ Fetched {len(documents)} items from API")
            return documents
            
        except Exception as e:
            logger.error(f"❌ API fetch failed: {e}")
            return []
    
    def process_and_embed(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process documents and generate embeddings.
        
        Args:
            documents: List of documents with 'text' and 'metadata'
            
        Returns:
            List of processed documents with embeddings
        """
        if not documents:
            return []
        
        logger.info(f"🔄 Processing {len(documents)} documents...")
        
        # Extract texts
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings in batch (much faster!)
        embeddings = self.embedding_service.generate_embeddings_batch(
            texts,
            show_progress=True
        )
        
        # Combine with documents
        processed_docs = []
        for doc, embedding in zip(documents, embeddings):
            processed_doc = {
                'text': doc['text'],
                'embedding': embedding,
                'metadata': doc.get('metadata', {}),
                'id': generate_id(doc['text']),
                'processed_at': get_timestamp()
            }
            processed_docs.append(processed_doc)
        
        logger.info(f"✅ Processed {len(processed_docs)} documents with embeddings")
        return processed_docs
    
    def update_database(
        self,
        collection_name: str,
        processed_docs: List[Dict[str, Any]]
    ) -> int:
        """
        Update vector database with processed documents.
        
        Args:
            collection_name: Name of collection to update
            processed_docs: List of processed documents with embeddings
            
        Returns:
            Number of documents added
        """
        if not processed_docs:
            logger.warning("⚠️ No documents to add")
            return 0
        
        logger.info(f"🔄 Updating database with {len(processed_docs)} documents...")
        
        try:
            # Prepare data for database
            texts = [doc['text'] for doc in processed_docs]
            embeddings = [doc['embedding'] for doc in processed_docs]
            metadatas = [doc['metadata'] for doc in processed_docs]
            ids = [doc['id'] for doc in processed_docs]
            
            # Add to database
            self.vector_db.add_documents(
                collection_name=collection_name,
                documents=texts,
                embeddings=embeddings,
                metadata=metadatas,
                ids=ids
            )
            
            logger.info(f"✅ Added {len(processed_docs)} documents to database")
            return len(processed_docs)
            
        except Exception as e:
            logger.error(f"❌ Database update failed: {e}")
            return 0
    
    def update_from_source(self, source_id: str) -> Dict[str, Any]:
        """
        Complete update cycle for a specific source.
        
        This is the main method that orchestrates the entire update process!
        
        Args:
            source_id: ID of the source to update from
            
        Returns:
            Dictionary with update results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 Starting update from source: {source_id}")
        logger.info(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Step 1: Fetch new content
        documents = self.fetch_new_content(source_id)
        
        if not documents:
            logger.warning(f"⚠️ No new content from {source_id}")
            return {
                'source_id': source_id,
                'success': False,
                'documents_added': 0,
                'error': 'No content fetched'
            }
        
        # Step 2: Process and embed
        processed_docs = self.process_and_embed(documents)
        
        # Step 3: Update database
        docs_added = self.update_database(self.collection_name, processed_docs)
        
        # Step 4: Update source metadata
        if source_id in self.sources:
            self.sources[source_id]['last_updated'] = get_timestamp()
            self.sources[source_id]['total_documents'] += docs_added
        
        # Step 5: Log update
        elapsed_time = time.time() - start_time
        update_log = {
            'source_id': source_id,
            'timestamp': get_timestamp(),
            'documents_added': docs_added,
            'elapsed_time': elapsed_time,
            'success': True
        }
        self.update_history.append(update_log)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Update complete!")
        logger.info(f"   Documents added: {docs_added}")
        logger.info(f"   Time taken: {elapsed_time:.2f}s")
        logger.info(f"{'='*60}\n")
        
        return update_log
    
    def update_all_sources(self) -> List[Dict[str, Any]]:
        """
        Update from all registered sources.
        
        Returns:
            List of update results for each source
        """
        logger.info(f"🔄 Updating from all {len(self.sources)} sources...")
        
        results = []
        for source_id in self.sources.keys():
            result = self.update_from_source(source_id)
            results.append(result)
            time.sleep(1)  # Be nice to servers
        
        total_docs = sum(r['documents_added'] for r in results)
        logger.info(f"✅ All sources updated! Total documents added: {total_docs}")
        
        return results
    
    def get_update_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent update logs.
        
        Args:
            limit: Maximum number of logs to return
            
        Returns:
            List of recent update logs
        """
        return self.update_history[-limit:]
    
    def get_source_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all sources.
        
        Returns:
            Dictionary with source statistics
        """
        stats = {
            'total_sources': len(self.sources),
            'total_updates': len(self.update_history),
            'sources': self.sources,
            'collection_stats': self.vector_db.get_collection_stats(self.collection_name)
        }
        return stats
    
    # ========================================================================
    # Scheduling Methods
    # ========================================================================
    
    def schedule_updates(
        self,
        interval: str = "daily",
        time_of_day: str = "00:00",
        source_ids: Optional[List[str]] = None
    ) -> None:
        """
        Schedule automatic updates at specified intervals.
        
        Intervals:
        - 'hourly': Every hour
        - 'daily': Once per day at specified time
        - 'weekly': Once per week at specified day and time
        - Custom: e.g., 'every 30 minutes'
        
        Args:
            interval: Update frequency
            time_of_day: Time to run (HH:MM format for daily/weekly)
            source_ids: List of source IDs to update (None = all sources)
        """
        logger.info(f"⏰ Setting up {interval} schedule...")
        
        # Define the update job
        def update_job():
            logger.info(f"\n{'='*60}")
            logger.info(f"⏰ Scheduled update triggered at {get_timestamp()}")
            logger.info(f"{'='*60}\n")
            
            try:
                if source_ids:
                    # Update specific sources
                    for source_id in source_ids:
                        self.update_from_source(source_id)
                else:
                    # Update all sources
                    self.update_all_sources()
                
                logger.info(f"✅ Scheduled update completed successfully")
                
            except Exception as e:
                logger.error(f"❌ Scheduled update failed: {e}")
        
        # Schedule based on interval
        if interval == "hourly":
            schedule.every().hour.do(update_job)
            logger.info(f"✅ Scheduled hourly updates")
            
        elif interval == "daily":
            schedule.every().day.at(time_of_day).do(update_job)
            logger.info(f"✅ Scheduled daily updates at {time_of_day}")
            
        elif interval == "weekly":
            # Default to Monday
            schedule.every().monday.at(time_of_day).do(update_job)
            logger.info(f"✅ Scheduled weekly updates on Monday at {time_of_day}")
            
        elif interval.startswith("every"):
            # Custom interval like "every 30 minutes"
            parts = interval.split()
            if len(parts) >= 3:
                amount = int(parts[1])
                unit = parts[2]
                
                if unit.startswith("minute"):
                    schedule.every(amount).minutes.do(update_job)
                elif unit.startswith("hour"):
                    schedule.every(amount).hours.do(update_job)
                elif unit.startswith("day"):
                    schedule.every(amount).days.do(update_job)
                
                logger.info(f"✅ Scheduled updates {interval}")
        
        else:
            logger.warning(f"⚠️ Unknown interval: {interval}")
    
    def start_scheduler(self, run_in_background: bool = True) -> Optional[threading.Thread]:
        """
        Start the scheduler to run scheduled updates.
        
        Args:
            run_in_background: If True, runs in a separate thread
            
        Returns:
            Thread object if running in background, None otherwise
        """
        logger.info("🚀 Starting scheduler...")
        
        def run_scheduler():
            logger.info("✅ Scheduler is running")
            logger.info("   Press Ctrl+C to stop")
            
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        if run_in_background:
            # Run in background thread
            scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            scheduler_thread.start()
            logger.info("✅ Scheduler started in background")
            return scheduler_thread
        else:
            # Run in foreground (blocking)
            run_scheduler()
            return None
    
    def stop_scheduler(self) -> None:
        """
        Stop all scheduled jobs.
        """
        schedule.clear()
        logger.info("🛑 Scheduler stopped - all jobs cleared")
    
    def get_scheduled_jobs(self) -> List[Dict[str, Any]]:
        """
        Get list of currently scheduled jobs.
        
        Returns:
            List of job information
        """
        jobs = []
        for job in schedule.get_jobs():
            jobs.append({
                'interval': str(job.interval),
                'unit': job.unit,
                'next_run': str(job.next_run) if job.next_run else None,
                'last_run': str(job.last_run) if job.last_run else None
            })
        
        logger.info(f"📋 Found {len(jobs)} scheduled jobs")
        return jobs


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Knowledge Base Updater with Scheduling\n")
    
    # Initialize components
    print("1️⃣ Initializing components...")
    vector_db = VectorDatabaseManager()
    embedding_service = EmbeddingService()
    updater = KnowledgeBaseUpdater(vector_db, embedding_service)
    
    # Add a file source (example)
    print("\n2️⃣ Adding test source...")
    updater.add_source(
        source_id="test_file",
        source_type="file",
        source_config={
            'path': './test_data.txt',
            'chunk_size': 500
        }
    )
    
    # Create test file
    print("\n3️⃣ Creating test data...")
    os.makedirs('.', exist_ok=True)
    with open('./test_data.txt', 'w') as f:
        f.write("""
        Machine learning is a subset of artificial intelligence.
        It focuses on building systems that learn from data.
        Deep learning uses neural networks with multiple layers.
        Natural language processing helps computers understand human language.
        """)
    
    # Update from source
    print("\n4️⃣ Updating knowledge base...")
    result = updater.update_from_source("test_file")
    print(f"   Result: {result}")
    
    # Get stats
    print("\n5️⃣ Getting statistics...")
    stats = updater.get_source_stats()
    print(f"   Total sources: {stats['total_sources']}")
    print(f"   Total updates: {stats['total_updates']}")
    print(f"   Documents in DB: {stats['collection_stats']['document_count']}")
    
    # Test scheduling
    print("\n6️⃣ Testing scheduling...")
    print("   Setting up daily schedule at 00:00...")
    updater.schedule_updates(interval="daily", time_of_day="00:00")
    
    print("   Setting up hourly schedule...")
    updater.schedule_updates(interval="hourly")
    
    # Get scheduled jobs
    jobs = updater.get_scheduled_jobs()
    print(f"   Scheduled jobs: {len(jobs)}")
    for i, job in enumerate(jobs, 1):
        print(f"     Job {i}: Every {job['interval']} {job['unit']}")
    
    print("\n   Note: Scheduler is configured but not started in test mode")
    print("   To start: updater.start_scheduler()")
    
    # Clean up
    updater.stop_scheduler()
    
    print("\n✅ All tests completed!")
