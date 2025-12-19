"""
Dataset caching utility to avoid reprocessing MedQuAD dataset every time.
Saves preprocessed Q&A pairs to disk for faster loading.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict


class DatasetCache:
    """Handles caching of preprocessed dataset"""
    
    def __init__(self, cache_dir: str = "data/medquad/cache"):
        """
        Initialize dataset cache.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "preprocessed_qa_pairs.pkl"
        self.stats_file = self.cache_dir / "dataset_stats.json"
    
    def save(self, qa_pairs: List[Dict], stats: Dict):
        """
        Save preprocessed Q&A pairs and statistics to cache.
        
        Args:
            qa_pairs: List of preprocessed Q&A pairs
            stats: Dataset statistics
        """
        try:
            # Save Q&A pairs
            with open(self.cache_file, 'wb') as f:
                pickle.dump(qa_pairs, f)
            
            # Save stats
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            
            print(f"✅ Cached {len(qa_pairs)} Q&A pairs to {self.cache_file}")
            
        except Exception as e:
            print(f"⚠️  Failed to save cache: {e}")
    
    def load(self) -> tuple:
        """
        Load preprocessed Q&A pairs and statistics from cache.
        
        Returns:
            Tuple of (qa_pairs, stats) or (None, None) if cache doesn't exist
        """
        try:
            if not self.cache_file.exists():
                return None, None
            
            # Load Q&A pairs
            with open(self.cache_file, 'rb') as f:
                qa_pairs = pickle.load(f)
            
            # Load stats
            stats = {}
            if self.stats_file.exists():
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
            
            print(f"✅ Loaded {len(qa_pairs)} Q&A pairs from cache")
            return qa_pairs, stats
            
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}")
            return None, None
    
    def exists(self) -> bool:
        """Check if cache exists"""
        return self.cache_file.exists()
    
    def clear(self):
        """Clear the cache"""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
            if self.stats_file.exists():
                self.stats_file.unlink()
            print("✅ Cache cleared")
        except Exception as e:
            print(f"⚠️  Failed to clear cache: {e}")
