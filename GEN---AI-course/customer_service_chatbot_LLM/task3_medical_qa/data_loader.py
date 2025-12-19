"""
MedQuAD Dataset Loader - Task 3
Downloads and preprocesses the MedQuAD (Medical Question Answering Dataset) from GitHub.
Parses XML files and extracts question-answer pairs for medical Q&A system.
"""

import os
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
import requests
import zipfile
import io
from pathlib import Path


class MedQuADDataLoader:
    """
    Handles downloading, parsing, and preprocessing of the MedQuAD dataset.
    
    The MedQuAD dataset contains medical question-answer pairs from various sources
    including NIH, CDC, and other medical organizations.
    """
    
    def __init__(self, data_dir: str = "data/medquad"):
        """
        Initialize the MedQuAD data loader.
        
        Args:
            data_dir: Directory to store the downloaded dataset
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.github_repo_url = "https://github.com/abachaa/MedQuAD"
        self.github_zip_url = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"
        self.raw_data_path = self.data_dir / "MedQuAD-master"
        
    def download_dataset(self) -> bool:
        """
        Download the MedQuAD dataset from GitHub.
        
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            print(f"Downloading MedQuAD dataset from {self.github_zip_url}...")
            
            # Check if already downloaded
            if self.raw_data_path.exists():
                print(f"Dataset already exists at {self.raw_data_path}")
                return True
            
            # Download the zip file
            response = requests.get(self.github_zip_url, timeout=300)
            response.raise_for_status()
            
            # Extract the zip file
            print("Extracting dataset...")
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            print(f"Dataset downloaded and extracted to {self.data_dir}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading dataset: {e}")
            return False
        except zipfile.BadZipFile as e:
            print(f"Error extracting zip file: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during download: {e}")
            return False
    
    def parse_xml_file(self, xml_path: Path) -> List[Dict[str, str]]:
        """
        Parse a single XML file and extract question-answer pairs.
        
        Args:
            xml_path: Path to the XML file
            
        Returns:
            List of dictionaries containing question-answer pairs with metadata
        """
        qa_pairs = []
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract document metadata
            source = root.get('source', 'Unknown')
            url = root.get('url', '')
            
            # Find all QAPairs in the document
            for qapair in root.findall('.//QAPair'):
                question_elem = qapair.find('Question')
                answer_elem = qapair.find('Answer')
                
                if question_elem is not None and answer_elem is not None:
                    question_text = question_elem.text
                    answer_text = answer_elem.text
                    
                    # Only add if both question and answer have content
                    if question_text and answer_text:
                        qa_pair = {
                            'question': question_text.strip(),
                            'answer': answer_text.strip(),
                            'source': source,
                            'url': url,
                            'file': xml_path.name
                        }
                        qa_pairs.append(qa_pair)
            
        except ET.ParseError as e:
            print(f"Error parsing XML file {xml_path}: {e}")
        except Exception as e:
            print(f"Unexpected error parsing {xml_path}: {e}")
        
        return qa_pairs
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize medical text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize common medical abbreviations spacing
        text = re.sub(r'\s*\.\s*', '. ', text)
        text = re.sub(r'\s*,\s*', ', ', text)
        
        # Remove multiple consecutive periods
        text = re.sub(r'\.{2,}', '.', text)
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Remove any remaining extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_medical_text(self, text: str) -> str:
        """
        Apply medical-specific text normalization.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized medical text
        """
        # First apply general cleaning
        text = self.clean_text(text)
        
        # Convert to lowercase for certain operations (but preserve original case)
        # This is mainly for consistency in medical terminology
        
        # Standardize common medical terms spacing
        text = re.sub(r'\bDr\.\s*', 'Dr. ', text)
        text = re.sub(r'\bMr\.\s*', 'Mr. ', text)
        text = re.sub(r'\bMrs\.\s*', 'Mrs. ', text)
        
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_all_qa_pairs(self) -> List[Dict[str, str]]:
        """
        Load and parse all XML files in the MedQuAD dataset.
        
        Returns:
            List of all question-answer pairs from the dataset
        """
        all_qa_pairs = []
        
        # Check if dataset exists
        if not self.raw_data_path.exists():
            print("Dataset not found. Downloading...")
            if not self.download_dataset():
                print("Failed to download dataset.")
                return []
        
        # Find all XML files in the dataset
        xml_files = list(self.raw_data_path.rglob("*.xml"))
        
        if not xml_files:
            print(f"No XML files found in {self.raw_data_path}")
            return []
        
        print(f"Found {len(xml_files)} XML files. Parsing...")
        
        # Parse each XML file
        for i, xml_file in enumerate(xml_files):
            qa_pairs = self.parse_xml_file(xml_file)
            all_qa_pairs.extend(qa_pairs)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(xml_files)} files...")
        
        print(f"Total question-answer pairs extracted: {len(all_qa_pairs)}")
        
        return all_qa_pairs
    
    def preprocess_qa_pairs(self, qa_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Clean and normalize all question-answer pairs.
        
        Args:
            qa_pairs: List of raw question-answer pairs
            
        Returns:
            List of cleaned and normalized question-answer pairs
        """
        processed_pairs = []
        
        print("Preprocessing question-answer pairs...")
        
        for qa_pair in qa_pairs:
            processed_pair = {
                'question': self.normalize_medical_text(qa_pair['question']),
                'answer': self.normalize_medical_text(qa_pair['answer']),
                'source': qa_pair['source'],
                'url': qa_pair['url'],
                'file': qa_pair['file']
            }
            
            # Only include pairs with non-empty question and answer after cleaning
            if processed_pair['question'] and processed_pair['answer']:
                processed_pairs.append(processed_pair)
        
        print(f"Preprocessing complete. {len(processed_pairs)} valid pairs.")
        
        return processed_pairs
    
    def get_dataset_statistics(self, qa_pairs: List[Dict[str, str]]) -> Dict[str, any]:
        """
        Calculate statistics about the dataset.
        
        Args:
            qa_pairs: List of question-answer pairs
            
        Returns:
            Dictionary containing dataset statistics
        """
        if not qa_pairs:
            return {}
        
        sources = {}
        total_questions = len(qa_pairs)
        
        for pair in qa_pairs:
            source = pair['source']
            sources[source] = sources.get(source, 0) + 1
        
        avg_question_length = sum(len(pair['question'].split()) for pair in qa_pairs) / total_questions
        avg_answer_length = sum(len(pair['answer'].split()) for pair in qa_pairs) / total_questions
        
        stats = {
            'total_pairs': total_questions,
            'sources': sources,
            'num_sources': len(sources),
            'avg_question_length': round(avg_question_length, 2),
            'avg_answer_length': round(avg_answer_length, 2)
        }
        
        return stats


def download_and_preprocess_medquad(data_dir: str = "data/medquad", use_cache: bool = True) -> Tuple[List[Dict[str, str]], Dict[str, any]]:
    """
    Convenience function to download and preprocess the MedQuAD dataset.
    Uses caching to avoid reprocessing on subsequent loads.
    
    Args:
        data_dir: Directory to store the dataset
        use_cache: If True, use cached data if available
        
    Returns:
        Tuple of (processed_qa_pairs, statistics)
    """
    from task3_medical_qa.dataset_cache import DatasetCache
    
    cache = DatasetCache(cache_dir=f"{data_dir}/cache")
    
    # Try to load from cache first
    if use_cache:
        cached_pairs, cached_stats = cache.load()
        if cached_pairs is not None:
            print("✅ Using cached dataset (fast load)")
            return cached_pairs, cached_stats
    
    print("📥 Loading dataset from source (this will take a few minutes)...")
    
    loader = MedQuADDataLoader(data_dir)
    
    # Download dataset
    loader.download_dataset()
    
    # Load all QA pairs
    qa_pairs = loader.load_all_qa_pairs()
    
    # Preprocess
    processed_pairs = loader.preprocess_qa_pairs(qa_pairs)
    
    # Get statistics
    stats = loader.get_dataset_statistics(processed_pairs)
    
    # Save to cache for next time
    if use_cache:
        cache.save(processed_pairs, stats)
    
    return processed_pairs, stats


if __name__ == "__main__":
    # Example usage
    print("MedQuAD Dataset Loader")
    print("=" * 50)
    
    # Download and preprocess
    qa_pairs, stats = download_and_preprocess_medquad()
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total Q&A pairs: {stats.get('total_pairs', 0)}")
    print(f"Number of sources: {stats.get('num_sources', 0)}")
    print(f"Average question length: {stats.get('avg_question_length', 0)} words")
    print(f"Average answer length: {stats.get('avg_answer_length', 0)} words")
    
    print("\nSources:")
    for source, count in stats.get('sources', {}).items():
        print(f"  {source}: {count} pairs")
    
    # Show a sample Q&A pair
    if qa_pairs:
        print("\nSample Q&A Pair:")
        print(f"Question: {qa_pairs[0]['question'][:200]}...")
        print(f"Answer: {qa_pairs[0]['answer'][:200]}...")
        print(f"Source: {qa_pairs[0]['source']}")
