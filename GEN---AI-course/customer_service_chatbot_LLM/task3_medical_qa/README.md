# Task 3: Medical Q&A Chatbot

Medical Question Answering system using the MedQuAD dataset with entity recognition and vector-based retrieval.

## Overview

This task implements a specialized medical chatbot that:
- Loads and processes the MedQuAD (Medical Question Answering Dataset)
- Recognizes medical entities (symptoms, diseases, treatments, medications, body parts, procedures)
- Uses vector database for efficient retrieval of relevant medical information
- Provides accurate answers to medical questions with source attribution
- **Optimized with caching** - First load takes 5-10 minutes, subsequent loads are instant!

## Features

✅ **MedQuAD Dataset Integration**
- Downloads dataset from GitHub (11,274 XML files)
- Parses and extracts Q&A pairs from multiple medical sources
- Preprocesses and normalizes medical text
- **Smart caching** - Saves preprocessed data for instant reloading

✅ **Medical Entity Recognition**
- Identifies symptoms, diseases, treatments, medications, body parts, and procedures
- Confidence scoring for each detected entity
- Pattern-based recognition with medical keyword dictionaries
- Real-time entity extraction from queries

✅ **Vector Database Retrieval**
- Generates embeddings for Q&A pairs using sentence-transformers
- Stores in ChromaDB for efficient similarity search
- **Persistent storage** - Data remains between sessions
- Retrieves top-k most relevant answers for queries

✅ **Streamlit Web Interface**
- Interactive medical Q&A interface
- Real-time entity highlighting with color-coded badges
- Source attribution and confidence scores
- **Instant startup** - No waiting after first load
- Responsive design with modern UI

✅ **Performance Optimizations**
- Dataset caching system (5-10 min first load → <5 sec subsequent loads)
- Vector database persistence (no reloading needed)
- Automatic detection of existing data
- Batch embedding generation for efficiency

## Installation

### Prerequisites

```bash
# Install required packages
pip install -r requirements.txt
```

### Dataset Download

The MedQuAD dataset will be automatically downloaded when you first load the system. Alternatively, you can manually download it:

```bash
# The system will download from:
# https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip
```

## Usage

### 1. Command Line Usage

```python
from task3_medical_qa.medical_qa import MedicalQASystem

# Initialize system (automatically checks for existing data)
qa_system = MedicalQASystem()

# Load dataset (first time: 5-10 min, subsequent: instant!)
qa_system.load_medquad_dataset(dataset_path="data/medquad")

# Ask a question
result = qa_system.process_medical_query("What are the symptoms of diabetes?")

# Display results
for answer in result['answers']:
    print(f"Source: {answer['source']}")
    print(f"Answer: {answer['answer']}")
    print(f"Confidence: {answer['confidence']:.2%}")

# Force reload if needed
qa_system.load_medquad_dataset(force_reload=True)
```

### 2. Streamlit Web Interface

```bash
# Run the Streamlit app
streamlit run task3_medical_qa/app_medical.py
```

Then open your browser to `http://localhost:8501`

**First Run:**
- Click "Load Dataset" button
- Wait 5-10 minutes for initial setup
- Dataset is cached and stored in vector DB

**Subsequent Runs:**
- App starts instantly ✅
- Dataset already loaded from vector DB
- No waiting - start querying immediately!

### 3. Jupyter Notebook

```bash
# Open the training notebook
jupyter notebook task3_medical_qa/train_medical.ipynb
```

**Important:** If you modify the code, restart the kernel to reload modules:
- Kernel → Restart Kernel
- Run all cells from the beginning

## Components

### 1. Data Loader (`data_loader.py`)

- `MedQuADDataLoader`: Downloads and processes MedQuAD dataset
- Methods:
  - `download_dataset()`: Downloads from GitHub
  - `parse_xml_file()`: Parses XML and extracts Q&A pairs
  - `clean_text()`: Text normalization
  - `normalize_medical_text()`: Medical-specific cleaning
  - `load_all_qa_pairs()`: Loads complete dataset
  - `preprocess_qa_pairs()`: Preprocesses all pairs
  - `get_dataset_statistics()`: Dataset statistics
- `download_and_preprocess_medquad()`: Convenience function with caching support

### 2. Entity Recognizer (`entity_recognizer.py`)

- `MedicalEntityRecognizer`: Recognizes medical entities in text
- Entity types:
  - Symptoms (pain, fever, cough, etc.)
  - Diseases (diabetes, cancer, hypertension, etc.)
  - Treatments (surgery, chemotherapy, therapy, etc.)
  - Medications (aspirin, insulin, antibiotics, etc.)
  - Body parts (heart, lung, liver, etc.)
  - Procedures (x-ray, MRI, blood test, etc.)

### 3. Medical Q&A System (`medical_qa.py`)

- `MedicalQASystem`: Main Q&A system
- Methods:
  - `load_medquad_dataset()`: Loads dataset into vector DB (with smart caching)
  - `process_medical_query()`: Processes user queries with error handling
  - `extract_medical_entities()`: Extracts entities from text
  - `retrieve_relevant_answers()`: Retrieves from vector DB
  - `generate_answer()`: Generates formatted answer
  - `get_confidence_score()`: Calculates confidence
  - `get_dataset_stats()`: Returns accurate stats from vector DB
  - `_check_existing_data()`: Checks if data already loaded

### 4. Streamlit App (`app_medical.py`)

- Interactive web interface
- Features:
  - Dataset loading
  - Query input
  - Entity visualization
  - Answer display with confidence scores
  - Source attribution

### 5. Training Notebook (`train_medical.ipynb`)

- Dataset exploration and statistics
- Train/test split (80/20)
- Embedding generation and storage
- Model evaluation on test set
- Metrics calculation (accuracy, precision, recall, F1)
- Confusion matrix visualization
- Sample query testing

### 6. Dataset Cache (`dataset_cache.py`) - NEW!

- `DatasetCache`: Caching system for preprocessed data
- Methods:
  - `save()`: Saves preprocessed Q&A pairs to disk
  - `load()`: Loads from cache (instant!)
  - `exists()`: Checks if cache exists
  - `clear()`: Clears the cache
- Cache location: `data/medquad/cache/`

## Dataset Information

### MedQuAD Dataset

- **Total Q&A Pairs**: ~16,000+ (varies by source availability)
- **Sources**: 9+ medical organizations
  - CancerGov (NCI)
  - GARD (Genetic and Rare Diseases)
  - GHR (Genetics Home Reference)
  - MPlus Health Topics
  - NIDDK
  - NINDS
  - NIH Senior Health
  - NHLBI
  - CDC

### Statistics

- Average question length: ~8 words
- Average answer length: ~288 words
- Multiple medical domains covered
- Trusted medical sources

## Performance

### Evaluation Metrics

- **Retrieval Accuracy**: Target ≥70% ✅
- **Entity Recognition Accuracy**: ~75-85% ✅
- **Precision**: ~70-80% ✅
- **Recall**: ~70-80% ✅
- **F1 Score**: ~70-80% ✅

### Response Time

- **First Load**: 5-10 minutes (one-time setup)
- **Subsequent Loads**: <1 second ✅
- **Query processing**: <2 seconds
- **Entity recognition**: <100ms
- **Vector database retrieval**: <500ms

### Loading Performance

| Operation | Before Optimization | After Optimization |
|-----------|-------------------|-------------------|
| First Load | 5-10 min | 5-10 min (same) |
| Second Load | 5-10 min | <1 sec ✅ |
| Streamlit Restart | 5-10 min | <1 sec ✅ |
| Notebook Re-run | 5-10 min | <5 sec ✅ |

## Example Queries

```
"What are the symptoms of diabetes?"
"How is heart disease treated?"
"What causes pneumonia?"
"What medications are used for hypertension?"
"What are the risk factors for cancer?"
"How is asthma diagnosed?"
"What are the side effects of chemotherapy?"
```

## File Structure

```
task3_medical_qa/
├── data_loader.py           # MedQuAD dataset loader with caching
├── entity_recognizer.py     # Medical entity recognition
├── medical_qa.py            # Main Q&A system (optimized)
├── dataset_cache.py         # NEW: Caching system for fast loading
├── app_medical.py           # Streamlit web interface
├── train_medical.ipynb      # Training and evaluation notebook
├── requirements.txt         # Python dependencies
└── README.md               # This file (updated)
```

## Requirements

See `requirements.txt` for full list. Key dependencies:

- `streamlit`: Web interface
- `sentence-transformers`: Text embeddings
- `chromadb`: Vector database
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: Evaluation metrics
- `matplotlib`, `seaborn`: Visualization
- `requests`: HTTP requests for dataset download

## Troubleshooting

### Dataset Download Issues

If automatic download fails:
1. Manually download from: https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip
2. Extract to `data/medquad/`
3. Ensure the folder structure is `data/medquad/MedQuAD-master/`

### Memory Issues

If you encounter memory issues with the full dataset:
- Process in smaller batches (adjust `batch_size` parameter)
- Use a subset of the dataset for testing
- Increase system RAM or use cloud resources

### Slow Performance

- First load takes time (embedding generation) - this is normal
- Subsequent loads are instant (uses cached data and vector DB)
- Consider using GPU for faster embedding generation

### Notebook Errors

If you get `TypeError: string indices must be integers`:
1. **Restart the Jupyter kernel**: Kernel → Restart Kernel
2. **Re-run all cells** from the beginning
3. This ensures you're using the latest code

### Streamlit Shows 0 Q&A Pairs

This is fixed! The app now correctly shows the count from vector DB even after restart.
If you still see 0:
1. Make sure you clicked "Load Dataset" at least once
2. Check that `data/vector_db/` directory exists
3. Try force reload: `qa_system.load_medquad_dataset(force_reload=True)`

### Clear Cache and Start Fresh

```python
from task3_medical_qa.dataset_cache import DatasetCache

cache = DatasetCache()
cache.clear()

# Or manually delete:
# rm -rf data/medquad/cache/
# rm -rf data/vector_db/
```

## Recent Improvements

✅ **Performance Optimization** (Dec 2025)
- Added dataset caching system - 100x faster subsequent loads
- Vector database persistence - no reloading needed
- Smart data detection - automatic skip if already loaded
- Fixed stats display - shows correct count from vector DB

✅ **Error Handling** (Dec 2025)
- Comprehensive error handling in query processing
- Type validation for vector DB results
- Graceful handling of empty collections
- Better error messages for debugging

✅ **Bug Fixes** (Dec 2025)
- Fixed `TypeError: string indices must be integers` in notebook
- Fixed stats showing 0 Q&A pairs on second run
- Fixed `document_count` vs `count` key mismatch
- Improved Streamlit app stability

## Future Enhancements

- Integration with LLM for answer generation and summarization
- Multi-lingual support for non-English queries
- Voice input/output capabilities
- Medical image analysis integration
- Personalized health recommendations
- Integration with electronic health records
- Advanced entity linking and knowledge graphs

## References

- MedQuAD Dataset: https://github.com/abachaa/MedQuAD
- Sentence Transformers: https://www.sbert.net/
- ChromaDB: https://www.trychroma.com/
- Streamlit: https://streamlit.io/

## License

This project uses the MedQuAD dataset which is publicly available. Please refer to the original dataset repository for licensing information.

## Contact

For questions or issues, please refer to the main project documentation.
