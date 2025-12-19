"""
Dataset Cache System for Domain Expert - Task 4
Implements caching to avoid reloading dataset every time.
Similar to Task 3 medical_qa cache system.
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class DatasetCache:
    """
    Manages caching of processed arXiv dataset to improve loading performance.
    """
    
    def __init__(self, cache_dir: str = "data/arxiv/cache"):
        """
        Initialize dataset cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.papers_cache_file = self.cache_dir / "papers_cache.pkl"
        self.metadata_cache_file = self.cache_dir / "cache_metadata.json"
        
        print(f"📁 Dataset cache initialized at: {self.cache_dir}")
    
    def _get_source_file_hash(self, file_path: str) -> str:
        """
        Calculate hash of source file to detect changes.
        
        Args:
            file_path: Path to source file
            
        Returns:
            MD5 hash of file content
        """
        if not os.path.exists(file_path):
            return ""
        
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_cache_metadata(self) -> Dict:
        """
        Load cache metadata.
        
        Returns:
            Cache metadata dictionary
        """
        if self.metadata_cache_file.exists():
            try:
                with open(self.metadata_cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading cache metadata: {e}")
        
        return {}
    
    def _save_cache_metadata(self, metadata: Dict):
        """
        Save cache metadata.
        
        Args:
            metadata: Metadata to save
        """
        try:
            with open(self.metadata_cache_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving cache metadata: {e}")
    
    def is_papers_cache_valid(self, source_file: str, max_age_hours: int = 24) -> bool:
        """
        Check if papers cache is valid and up-to-date.
        
        Args:
            source_file: Path to source processed papers file
            max_age_hours: Maximum cache age in hours
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not self.papers_cache_file.exists():
            return False
        
        metadata = self._get_cache_metadata()
        
        # Check if metadata exists
        if 'papers_cache' not in metadata:
            return False
        
        cache_info = metadata['papers_cache']
        
        # Check file hash
        current_hash = self._get_source_file_hash(source_file)
        if cache_info.get('source_hash') != current_hash:
            print("📄 Source file changed, cache invalid")
            return False
        
        # Check age
        cache_time = datetime.fromisoformat(cache_info.get('created_at', ''))
        age = datetime.now() - cache_time
        if age > timedelta(hours=max_age_hours):
            print(f"⏰ Cache expired (age: {age})")
            return False
        
        print(f"✅ Papers cache is valid (age: {age})")
        return True
    
    def save_papers_cache(self, papers: List[Dict], source_file: str):
        """
        Save papers to cache.
        
        Args:
            papers: List of processed papers
            source_file: Path to source file
        """
        try:
            print(f"💾 Saving {len(papers)} papers to cache...")
            
            # Save papers data
            with open(self.papers_cache_file, 'wb') as f:
                pickle.dump(papers, f)
            
            # Update metadata
            metadata = self._get_cache_metadata()
            metadata['papers_cache'] = {
                'created_at': datetime.now().isoformat(),
                'source_file': source_file,
                'source_hash': self._get_source_file_hash(source_file),
                'paper_count': len(papers),
                'cache_size_mb': os.path.getsize(self.papers_cache_file) / (1024 * 1024)
            }
            self._save_cache_metadata(metadata)
            
            print(f"✅ Papers cache saved successfully")
            
        except Exception as e:
            print(f"❌ Error saving papers cache: {e}")
    
    def load_papers_cache(self) -> Optional[List[Dict]]:
        """
        Load papers from cache.
        
        Returns:
            List of cached papers or None if cache invalid
        """
        try:
            if not self.papers_cache_file.exists():
                return None
            
            print("📥 Loading papers from cache...")
            
            with open(self.papers_cache_file, 'rb') as f:
                papers = pickle.load(f)
            
            print(f"✅ Loaded {len(papers)} papers from cache")
            return papers
            
        except Exception as e:
            print(f"❌ Error loading papers cache: {e}")
            return None
    
    def is_vector_db_loaded(self, collection_name: str) -> bool:
        """
        Check if vector database has been loaded for this collection.
        
        Args:
            collection_name: Name of the vector database collection
            
        Returns:
            True if vector DB is loaded, False otherwise
        """
        flag_file = self.cache_dir / f"vector_db_{collection_name}.flag"
        
        if not flag_file.exists():
            return False
        
        try:
            with open(flag_file, 'r') as f:
                flag_data = json.load(f)
            
            # Check if flag is recent (within 24 hours)
            flag_time = datetime.fromisoformat(flag_data.get('created_at', ''))
            age = datetime.now() - flag_time
            
            if age > timedelta(hours=24):
                print(f"⏰ Vector DB flag expired (age: {age})")
                return False
            
            print(f"✅ Vector DB already loaded for {collection_name}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error reading vector DB flag: {e}")
            return False
    
    def mark_vector_db_loaded(self, collection_name: str, paper_count: int):
        """
        Mark vector database as loaded for this collection.
        
        Args:
            collection_name: Name of the vector database collection
            paper_count: Number of papers loaded
        """
        try:
            flag_file = self.cache_dir / f"vector_db_{collection_name}.flag"
            
            flag_data = {
                'created_at': datetime.now().isoformat(),
                'collection_name': collection_name,
                'paper_count': paper_count,
                'status': 'loaded'
            }
            
            with open(flag_file, 'w') as f:
                json.dump(flag_data, f, indent=2)
            
            print(f"✅ Marked vector DB as loaded: {collection_name}")
            
        except Exception as e:
            print(f"❌ Error marking vector DB as loaded: {e}")
    
    def clear_cache(self):
        """
        Clear all cache files.
        """
        try:
            cache_files = [
                self.papers_cache_file,
                self.metadata_cache_file
            ]
            
            # Clear vector DB flags
            for flag_file in self.cache_dir.glob("vector_db_*.flag"):
                cache_files.append(flag_file)
            
            cleared_count = 0
            for cache_file in cache_files:
                if cache_file.exists():
                    cache_file.unlink()
                    cleared_count += 1
            
            print(f"🗑️ Cleared {cleared_count} cache files")
            
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
    
    def get_cache_info(self) -> Dict:
        """
        Get information about current cache status.
        
        Returns:
            Dictionary with cache information
        """
        info = {
            'cache_dir': str(self.cache_dir),
            'papers_cache_exists': self.papers_cache_file.exists(),
            'metadata_exists': self.metadata_cache_file.exists(),
            'cache_files': []
        }
        
        # Get cache file sizes
        for cache_file in self.cache_dir.glob("*"):
            if cache_file.is_file():
                info['cache_files'].append({
                    'name': cache_file.name,
                    'size_mb': cache_file.stat().st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(cache_file.stat().st_mtime).isoformat()
                })
        
        # Get metadata
        metadata = self._get_cache_metadata()
        if metadata:
            info['metadata'] = metadata
        
        return info