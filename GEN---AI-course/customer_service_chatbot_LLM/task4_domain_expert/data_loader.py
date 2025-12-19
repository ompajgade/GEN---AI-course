"""
arXiv Dataset Loader - Task 4
Downloads and preprocesses arXiv scientific papers dataset.
Filters by domain and extracts paper metadata for domain expert system.
"""

import os
import json
import requests
import gzip
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
import re


class ArxivDataLoader:
    """
    Handles downloading, parsing, and preprocessing of arXiv dataset.
    
    The arXiv dataset contains scientific papers with metadata including
    title, abstract, categories, authors, and submission dates.
    """
    
    def __init__(self, data_dir: str = "data/arxiv"):
        """
        Initialize the arXiv data loader.
        
        Args:
            data_dir: Directory to store the downloaded dataset
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # arXiv API and Kaggle dataset URLs
        self.arxiv_api_url = "http://export.arxiv.org/api/query"
        self.kaggle_dataset_url = "https://www.kaggle.com/datasets/Cornell-University/arxiv"
        
        # Computer Science categories we're interested in
        self.cs_categories = {
            'cs.AI': 'Artificial Intelligence',
            'cs.LG': 'Machine Learning',
            'cs.CL': 'Computation and Language',
            'cs.CV': 'Computer Vision and Pattern Recognition',
            'cs.NE': 'Neural and Evolutionary Computing',
            'cs.RO': 'Robotics',
            'cs.IR': 'Information Retrieval',
            'cs.HC': 'Human-Computer Interaction',
            'cs.DS': 'Data Structures and Algorithms',
            'cs.DB': 'Databases'
        }
        
        self.raw_data_file = self.data_dir / "arxiv_papers.json"
        self.processed_data_file = self.data_dir / "processed_papers.json"
    
    def download_sample_papers(self, max_papers: int = 10000, categories: List[str] = None) -> bool:
        """
        Download sample papers from arXiv API.
        
        Args:
            max_papers: Maximum number of papers to download
            categories: List of categories to filter by (default: CS categories)
            
        Returns:
            True if successful, False otherwise
        """
        if categories is None:
            categories = list(self.cs_categories.keys())
        
        print(f"📥 Downloading arXiv papers for categories: {categories}")
        print(f"📊 Target: {max_papers} papers")
        
        all_papers = []
        papers_per_category = max_papers // len(categories)
        
        for category in categories:
            print(f"\n🔍 Fetching papers from category: {category} ({self.cs_categories.get(category, category)})")
            
            # Query arXiv API for this category
            query_params = {
                'search_query': f'cat:{category}',
                'start': 0,
                'max_results': papers_per_category,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            try:
                response = requests.get(self.arxiv_api_url, params=query_params, timeout=30)
                response.raise_for_status()
                
                # Parse XML response
                papers = self._parse_arxiv_xml(response.text, category)
                all_papers.extend(papers)
                print(f"✅ Downloaded {len(papers)} papers from {category}")
                
            except Exception as e:
                print(f"❌ Error downloading from {category}: {str(e)}")
                continue
        
        if all_papers:
            # Save raw data
            with open(self.raw_data_file, 'w', encoding='utf-8') as f:
                json.dump(all_papers, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Successfully downloaded {len(all_papers)} papers")
            print(f"💾 Saved to: {self.raw_data_file}")
            return True
        else:
            print("❌ No papers downloaded")
            return False
    
    def _parse_arxiv_xml(self, xml_content: str, category: str) -> List[Dict]:
        """
        Parse arXiv API XML response to extract paper information.
        
        Args:
            xml_content: XML response from arXiv API
            category: Category being processed
            
        Returns:
            List of paper dictionaries
        """
        import xml.etree.ElementTree as ET
        
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            entries = root.findall('atom:entry', namespaces)
            
            for entry in entries:
                try:
                    # Extract paper ID
                    paper_id = entry.find('atom:id', namespaces).text.split('/')[-1]
                    
                    # Extract title
                    title = entry.find('atom:title', namespaces).text.strip()
                    title = re.sub(r'\s+', ' ', title)  # Clean whitespace
                    
                    # Extract abstract
                    summary = entry.find('atom:summary', namespaces).text.strip()
                    summary = re.sub(r'\s+', ' ', summary)  # Clean whitespace
                    
                    # Extract authors
                    authors = []
                    for author in entry.findall('atom:author', namespaces):
                        name = author.find('atom:name', namespaces)
                        if name is not None:
                            authors.append(name.text.strip())
                    
                    # Extract categories
                    categories = []
                    for cat in entry.findall('atom:category', namespaces):
                        term = cat.get('term')
                        if term:
                            categories.append(term)
                    
                    # Extract published date
                    published = entry.find('atom:published', namespaces).text
                    
                    # Extract PDF link
                    pdf_link = None
                    for link in entry.findall('atom:link', namespaces):
                        if link.get('title') == 'pdf':
                            pdf_link = link.get('href')
                            break
                    
                    paper = {
                        'id': paper_id,
                        'title': title,
                        'abstract': summary,
                        'authors': authors,
                        'categories': categories,
                        'primary_category': category,
                        'published_date': published,
                        'pdf_link': pdf_link,
                        'full_text': '',  # Will be populated if needed
                        'processed_date': datetime.now().isoformat()
                    }
                    
                    papers.append(paper)
                    
                except Exception as e:
                    print(f"⚠️ Error parsing entry: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error parsing XML: {str(e)}")
        
        return papers
    
    def load_kaggle_dataset(self, dataset_path: str = None) -> bool:
        """
        Load arXiv dataset from Kaggle (if available locally).
        
        Args:
            dataset_path: Path to Kaggle dataset JSON file
            
        Returns:
            True if successful, False otherwise
        """
        if dataset_path is None:
            dataset_path = self.data_dir / "arxiv-metadata-oai-snapshot.json"
        
        if not os.path.exists(dataset_path):
            print(f"❌ Kaggle dataset not found at: {dataset_path}")
            print("💡 Please download from: https://www.kaggle.com/datasets/Cornell-University/arxiv")
            return False
        
        print(f"📥 Loading Kaggle arXiv dataset from: {dataset_path}")
        
        papers = []
        cs_papers_count = 0
        total_papers = 0
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    total_papers += 1
                    
                    if total_papers % 10000 == 0:
                        print(f"📊 Processed {total_papers} papers, found {cs_papers_count} CS papers")
                    
                    try:
                        paper_data = json.loads(line.strip())
                        
                        # Filter for computer science papers
                        categories = paper_data.get('categories', '').split()
                        if any(cat in self.cs_categories for cat in categories):
                            
                            # Extract relevant information
                            paper = {
                                'id': paper_data.get('id', ''),
                                'title': paper_data.get('title', '').strip(),
                                'abstract': paper_data.get('abstract', '').strip(),
                                'authors': paper_data.get('authors_parsed', []),
                                'categories': categories,
                                'primary_category': categories[0] if categories else '',
                                'published_date': paper_data.get('versions', [{}])[-1].get('created', ''),
                                'pdf_link': f"https://arxiv.org/pdf/{paper_data.get('id', '')}.pdf",
                                'full_text': '',
                                'processed_date': datetime.now().isoformat()
                            }
                            
                            papers.append(paper)
                            cs_papers_count += 1
                            
                            # Limit to manageable size
                            if cs_papers_count >= 10000:
                                break
                                
                    except json.JSONDecodeError:
                        continue
            
            if papers:
                # Save processed data
                with open(self.raw_data_file, 'w', encoding='utf-8') as f:
                    json.dump(papers, f, indent=2, ensure_ascii=False)
                
                print(f"\n✅ Successfully loaded {len(papers)} CS papers from Kaggle dataset")
                print(f"💾 Saved to: {self.raw_data_file}")
                return True
            else:
                print("❌ No CS papers found in dataset")
                return False
                
        except Exception as e:
            print(f"❌ Error loading Kaggle dataset: {str(e)}")
            return False
    
    def preprocess_papers(self, min_abstract_length: int = 100, max_papers: int = 5000) -> bool:
        """
        Preprocess and clean the downloaded papers.
        
        Args:
            min_abstract_length: Minimum abstract length to keep paper
            max_papers: Maximum number of papers to keep after preprocessing
            
        Returns:
            True if successful, False otherwise
        """
        if not self.raw_data_file.exists():
            print("❌ No raw data file found. Please download papers first.")
            return False
        
        print("🔄 Preprocessing arXiv papers...")
        
        with open(self.raw_data_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        print(f"📊 Starting with {len(papers)} papers")
        
        # Filter and clean papers
        processed_papers = []
        
        for paper in papers:
            # Skip papers with short abstracts
            if len(paper.get('abstract', '')) < min_abstract_length:
                continue
            
            # Clean title and abstract
            title = self._clean_text(paper.get('title', ''))
            abstract = self._clean_text(paper.get('abstract', ''))
            
            if not title or not abstract:
                continue
            
            # Create processed paper
            processed_paper = {
                'id': paper.get('id', ''),
                'title': title,
                'abstract': abstract,
                'authors': paper.get('authors', []),
                'categories': paper.get('categories', []),
                'primary_category': paper.get('primary_category', ''),
                'category_name': self.cs_categories.get(paper.get('primary_category', ''), 'Unknown'),
                'published_date': paper.get('published_date', ''),
                'pdf_link': paper.get('pdf_link', ''),
                'text_for_embedding': f"{title} {abstract}",  # Combined text for embeddings
                'processed_date': datetime.now().isoformat()
            }
            
            processed_papers.append(processed_paper)
            
            # Limit number of papers
            if len(processed_papers) >= max_papers:
                break
        
        if processed_papers:
            # Save processed data
            with open(self.processed_data_file, 'w', encoding='utf-8') as f:
                json.dump(processed_papers, f, indent=2, ensure_ascii=False)
            
            # Generate statistics
            self._generate_statistics(processed_papers)
            
            print(f"\n✅ Successfully preprocessed {len(processed_papers)} papers")
            print(f"💾 Saved to: {self.processed_data_file}")
            return True
        else:
            print("❌ No papers remaining after preprocessing")
            return False
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove LaTeX commands (basic cleanup)
        text = re.sub(r'\$[^$]*\$', '', text)  # Remove inline math
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)  # Remove LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+', '', text)  # Remove remaining LaTeX commands
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _generate_statistics(self, papers: List[Dict]) -> None:
        """
        Generate and display statistics about the processed papers.
        
        Args:
            papers: List of processed papers
        """
        print("\n📈 Dataset Statistics:")
        print(f"   Total papers: {len(papers)}")
        
        # Category distribution
        category_counts = {}
        for paper in papers:
            cat = paper.get('primary_category', 'Unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print("\n📊 Category Distribution:")
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            cat_name = self.cs_categories.get(cat, cat)
            print(f"   {cat} ({cat_name}): {count} papers")
        
        # Abstract length statistics
        abstract_lengths = [len(paper.get('abstract', '')) for paper in papers]
        if abstract_lengths:
            print(f"\n📝 Abstract Length Statistics:")
            print(f"   Average: {sum(abstract_lengths) / len(abstract_lengths):.0f} characters")
            print(f"   Min: {min(abstract_lengths)} characters")
            print(f"   Max: {max(abstract_lengths)} characters")
    
    def load_processed_papers(self) -> Optional[List[Dict]]:
        """
        Load preprocessed papers from file.
        
        Returns:
            List of processed papers or None if file doesn't exist
        """
        if not self.processed_data_file.exists():
            print("❌ No processed data file found. Please preprocess papers first.")
            return None
        
        try:
            with open(self.processed_data_file, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            
            print(f"✅ Loaded {len(papers)} processed papers")
            return papers
            
        except Exception as e:
            print(f"❌ Error loading processed papers: {str(e)}")
            return None
    
    def get_papers_by_category(self, category: str) -> List[Dict]:
        """
        Get papers filtered by category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of papers in the specified category
        """
        papers = self.load_processed_papers()
        if not papers:
            return []
        
        filtered_papers = [
            paper for paper in papers 
            if paper.get('primary_category') == category or category in paper.get('categories', [])
        ]
        
        print(f"🔍 Found {len(filtered_papers)} papers in category: {category}")
        return filtered_papers


def main():
    """
    Main function to demonstrate the arXiv data loader.
    """
    print("🚀 arXiv Dataset Loader - Task 4")
    print("=" * 50)
    
    # Initialize loader
    loader = ArxivDataLoader()
    
    # Try to load from Kaggle dataset first (if available)
    kaggle_path = "data/arxiv/arxiv-metadata-oai-snapshot.json"
    if os.path.exists(kaggle_path):
        print("📁 Found Kaggle dataset, loading...")
        if loader.load_kaggle_dataset(kaggle_path):
            loader.preprocess_papers()
        else:
            print("❌ Failed to load Kaggle dataset")
    else:
        # Download sample papers from API
        print("🌐 Downloading sample papers from arXiv API...")
        if loader.download_sample_papers(max_papers=1000):
            loader.preprocess_papers()
        else:
            print("❌ Failed to download papers")
    
    # Load and display statistics
    papers = loader.load_processed_papers()
    if papers:
        print(f"\n✅ Dataset ready with {len(papers)} papers")
        
        # Show sample paper
        if papers:
            sample = papers[0]
            print(f"\n📄 Sample Paper:")
            print(f"   Title: {sample['title'][:100]}...")
            print(f"   Category: {sample['primary_category']} ({sample['category_name']})")
            print(f"   Abstract: {sample['abstract'][:200]}...")


if __name__ == "__main__":
    main()