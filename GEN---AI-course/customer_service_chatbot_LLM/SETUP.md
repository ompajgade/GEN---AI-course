# GenAI Customer Service Bot - Course Curriculum Project

## Overview

This project implements a comprehensive GenAI Customer Service Bot system as part of the course curriculum. It includes 6 integrated tasks that work together to provide advanced AI-powered customer service capabilities.

## Project Structure

```
genai-customer-service-bot/
├── task1_knowledge_updater/    # Task 1: Dynamic Knowledge Base Expansion
│   ├── app_updater.py         # Streamlit interface
│   ├── knowledge_updater.py   # Core functionality
│   └── README.md              # Task-specific documentation
├── task2_multimodal/           # Task 2: Multi-Modal Chatbot
├── task3_medical_qa/           # Task 3: Medical Q&A System
│   ├── app_medical.py         # Streamlit interface
│   ├── medical_qa.py          # Core functionality
│   └── README.md              # Task-specific documentation
├── task4_domain_expert/        # Task 4: Domain Expert System (arXiv)
│   ├── app_expert.py          # Streamlit interface
│   ├── domain_expert.py       # Core functionality
│   ├── data_loader.py         # Dataset management
│   └── README.md              # Task-specific documentation
├── task5_sentiment/            # Task 5: Sentiment Analysis Integration
├── task6_multilingual/         # Task 6: Multi-Lingual Support
├── shared/                     # Shared components across all tasks
│   ├── vector_db_manager.py   # Vector database management
│   ├── embedding_service.py   # Embedding generation
│   ├── llm_integration.py     # LLM API integration
│   ├── evaluation.py          # Evaluation framework
│   └── utils.py               # Utility functions
├── data/                       # Datasets (excluded from git)
├── models/                     # Model storage (excluded from git)
├── evaluation_results/         # Evaluation outputs
├── tests/                      # Comprehensive test suite
├── app.py                      # Main application entry point
├── run_evaluation.py          # Evaluation runner
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore configuration
└── README.md                  # Main project documentation
```

## Quick Start

### 1. Environment Setup

**Create Virtual Environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

### 2. Configuration

**Set up Environment Variables:**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys:
# GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### 3. Running the Applications

**Main Application:**
```bash
streamlit run app.py
```

**Individual Task Applications:**
```bash
# Task 1: Knowledge Base Updater
streamlit run task1_knowledge_updater/app_updater.py

# Task 3: Medical Q&A
streamlit run task3_medical_qa/app_medical.py

# Task 4: Domain Expert System
streamlit run task4_domain_expert/app_expert.py
```

## Task Descriptions

### Task 1: Dynamic Knowledge Base Expansion
- **Purpose**: Automatically expand knowledge base from various sources
- **Features**: RSS feeds, web scraping, file uploads, scheduled updates
- **Interface**: `task1_knowledge_updater/app_updater.py`

### Task 2: Multi-Modal Chatbot
- **Purpose**: Handle text and image inputs for customer service
- **Features**: Image analysis, multi-modal responses
- **Interface**: `task2_multimodal/` (implementation in progress)

### Task 3: Medical Q&A System
- **Purpose**: Provide medical information using MedQuAD dataset
- **Features**: Entity recognition, vector search, confidence scoring
- **Interface**: `task3_medical_qa/app_medical.py`

### Task 4: Domain Expert System
- **Purpose**: Scientific paper search and explanation using arXiv
- **Features**: Paper search, summarization, concept explanation, expert chat
- **Interface**: `task4_domain_expert/app_expert.py`

### Task 5: Sentiment Analysis Integration
- **Purpose**: Analyze customer sentiment and adjust responses
- **Features**: Real-time sentiment detection, response adaptation
- **Interface**: `task5_sentiment/` (implementation in progress)

### Task 6: Multi-Lingual Support
- **Purpose**: Support multiple languages for global customer service
- **Features**: Language detection, translation, localized responses
- **Interface**: `task6_multilingual/` (implementation in progress)

## Evaluation

**Run Comprehensive Evaluation:**
```bash
python run_evaluation.py
```

**Evaluation Results:**
- Results are saved in `evaluation_results/`
- Includes performance metrics, accuracy scores, and detailed analysis

## API Keys Required

- **Google Gemini API**: For LLM functionality (required)
- **Hugging Face Token**: For some models (optional)

## Dependencies

All dependencies are listed in `requirements.txt` and include:
- Streamlit for web interfaces
- Google Generative AI for LLM integration
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Various ML and data processing libraries

## Testing

**Run Tests:**
```bash
pytest tests/
```

**Test Coverage:**
- Unit tests for individual components
- Integration tests for task workflows
- Property-based tests for correctness validation

## Submission Notes

This project is designed as an integrated system where all tasks work together within the course curriculum framework. Each task is organized in its own folder but shares common components through the `shared/` directory.

## Support

For issues or questions, refer to the individual task README files or the main project documentation.
