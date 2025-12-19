"""
LLM Integration Layer
Provides unified interface to Google Gemini and Palm APIs.
Handles text generation, multi-modal processing, and embeddings.
"""

import google.generativeai as genai
from typing import List, Optional, Dict, Any
import os
import time
import logging
from PIL import Image
import io
import base64
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMIntegration:
    """
    Unified interface for Large Language Model operations.
    
    Key Features:
    - Text generation (conversations, Q&A, explanations)
    - Multi-modal processing (text + images)
    - Image analysis and generation
    - Retry logic for API failures
    - Fallback mechanisms
    """
    
    def __init__(
        self,
        primary_model: str = "gemini-pro",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the LLM integration.
        
        Args:
            primary_model: Primary model to use (gemini-pro, gemini-pro-vision)
            api_key: Google API key (reads from env if not provided)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.primary_model = primary_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        
        if not self.api_key:
            logger.warning("⚠️ No Google API key found. Set GOOGLE_API_KEY environment variable.")
            logger.warning("   Some features will not work without an API key.")
        else:
            # Configure the API
            genai.configure(api_key=self.api_key)
            logger.info(f"✅ LLM Integration initialized with model: {primary_model}")
        
        # Initialize models
        self.text_model = None
        self.vision_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the generative models."""
        try:
            if self.api_key:
                # Use the latest Gemini model names (gemini-2.5-flash supports both text and vision)
                self.text_model = genai.GenerativeModel('gemini-2.5-flash')
                self.vision_model = genai.GenerativeModel('gemini-2.5-flash')
                logger.info("✅ Models initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize models: {e}")
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute a function with exponential backoff retry logic.
        
        This handles temporary API failures gracefully.
        """
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}")
                    logger.info(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ All {self.max_retries} attempts failed")
                    raise
    
    def generate_text(
        self,
        prompt: str,
        context: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text response using the LLM.
        
        This is the core function for generating AI responses!
        
        Args:
            prompt: The user's question or prompt
            context: Additional context to help the model
            max_tokens: Maximum length of response
            temperature: Creativity level (0.0 = focused, 1.0 = creative)
            
        Returns:
            Generated text response
        """
        if not self.text_model:
            return "Error: LLM not initialized. Please set GOOGLE_API_KEY."
        
        try:
            # Combine context and prompt
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            # Generate response with retry logic
            def _generate():
                response = self.text_model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature
                    )
                )
                return response.text
            
            result = self._retry_with_backoff(_generate)
            logger.info(f"✅ Generated text response ({len(result)} chars)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Text generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_multimodal(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate response from text and/or image input.
        
        This enables the multi-modal chatbot (Task 2)!
        
        Args:
            prompt: Text prompt or question
            image: Optional PIL Image object
            max_tokens: Maximum response length
            
        Returns:
            Generated response
        """
        if not self.vision_model:
            return "Error: Vision model not initialized. Please set GOOGLE_API_KEY."
        
        try:
            def _generate():
                if image:
                    # Multi-modal: text + image
                    response = self.vision_model.generate_content(
                        [prompt, image],
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=max_tokens
                        )
                    )
                else:
                    # Text only
                    response = self.text_model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=max_tokens
                        )
                    )
                return response.text
            
            result = self._retry_with_backoff(_generate)
            logger.info(f"✅ Generated multi-modal response")
            return result
            
        except Exception as e:
            logger.error(f"❌ Multi-modal generation failed: {e}")
            return f"Error: {str(e)}"
    
    def analyze_image(self, image: Image.Image, question: str = "Describe this image in detail.") -> Dict[str, Any]:
        """
        Analyze an image and return structured information.
        
        Args:
            image: PIL Image object
            question: What to ask about the image
            
        Returns:
            Dictionary with analysis results
        """
        if not self.vision_model:
            return {"error": "Vision model not initialized"}
        
        try:
            description = self.generate_multimodal(question, image)
            
            return {
                "description": description,
                "image_size": image.size,
                "image_mode": image.mode,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Image analysis failed: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    def get_embeddings(self, text: str, model: str = "models/embedding-001") -> List[float]:
        """
        Generate embeddings for text.
        
        Embeddings are numerical representations used by the vector database.
        
        Args:
            text: Text to embed
            model: Embedding model to use
            
        Returns:
            List of floats representing the embedding
        """
        try:
            def _embed():
                result = genai.embed_content(
                    model=model,
                    content=text,
                    task_type="retrieval_document"
                )
                return result['embedding']
            
            embedding = self._retry_with_backoff(_embed)
            logger.info(f"✅ Generated embedding (dimension: {len(embedding)})")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            # Return a zero vector as fallback
            return [0.0] * 768  # Standard embedding dimension
    
    def generate_with_context(
        self,
        query: str,
        retrieved_documents: List[str],
        system_prompt: str = "You are a helpful AI assistant."
    ) -> str:
        """
        Generate response using retrieved context from vector database.
        
        This is the RAG (Retrieval Augmented Generation) pattern!
        
        Args:
            query: User's question
            retrieved_documents: Relevant documents from vector DB
            system_prompt: System instructions for the model
            
        Returns:
            Generated response
        """
        # Build context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1}:\n{doc}"
            for i, doc in enumerate(retrieved_documents)
        ])
        
        # Create full prompt
        full_prompt = f"""{system_prompt}

Context Information:
{context}

User Question: {query}

Please provide a helpful and accurate response based on the context provided."""
        
        return self.generate_text(full_prompt)
    
    def summarize_text(
        self,
        text: str,
        max_length: int = 200,
        style: str = "concise"
    ) -> str:
        """
        Summarize long text into shorter form.
        
        Used by Task 4 (Domain Expert) for paper summaries.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in words
            style: Summary style (concise, detailed, bullet-points)
            
        Returns:
            Summarized text
        """
        prompt = f"""Summarize the following text in a {style} style, 
keeping it under {max_length} words:

{text}

Summary:"""
        
        return self.generate_text(prompt, max_tokens=max_length * 2)
    
    def explain_concept(
        self,
        concept: str,
        context: str = "",
        audience: str = "general"
    ) -> str:
        """
        Explain a complex concept in understandable terms.
        
        Used by Task 4 (Domain Expert) for explaining scientific concepts.
        
        Args:
            concept: The concept to explain
            context: Additional context
            audience: Target audience (general, technical, beginner)
            
        Returns:
            Explanation text
        """
        prompt = f"""Explain the concept of "{concept}" for a {audience} audience.
        
{f"Context: {context}" if context else ""}

Provide a clear, accurate explanation with examples if helpful."""
        
        return self.generate_text(prompt)
    
    def adjust_tone(
        self,
        text: str,
        target_sentiment: str,
        original_sentiment: str = "neutral"
    ) -> str:
        """
        Adjust the tone of text based on sentiment.
        
        Used by Task 5 (Sentiment Analysis) for empathetic responses.
        
        Args:
            text: Original text
            target_sentiment: Desired sentiment (positive, negative, neutral)
            original_sentiment: Current sentiment
            
        Returns:
            Text with adjusted tone
        """
        prompt = f"""Rewrite the following text to have a {target_sentiment} tone,
while keeping the core message intact:

Original text: {text}

Rewritten text:"""
        
        return self.generate_text(prompt, temperature=0.5)
    
    def translate_and_adapt(
        self,
        text: str,
        target_language: str,
        cultural_context: str = ""
    ) -> str:
        """
        Translate text and adapt it culturally.
        
        Used by Task 6 (Multi-lingual Support).
        
        Args:
            text: Text to translate
            target_language: Target language code
            cultural_context: Cultural adaptation notes
            
        Returns:
            Translated and adapted text
        """
        prompt = f"""Translate the following text to {target_language}.
Make it culturally appropriate and natural-sounding.

{f"Cultural context: {cultural_context}" if cultural_context else ""}

Text to translate: {text}

Translation:"""
        
        return self.generate_text(prompt)


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing LLM Integration\n")
    
    # Initialize LLM
    llm = LLMIntegration()
    
    # Test 1: Simple text generation
    print("1️⃣ Testing text generation...")
    response = llm.generate_text(
        prompt="What is machine learning?",
        max_tokens=100
    )
    print(f"   Response: {response[:100]}...\n")
    
    # Test 2: Context-based generation
    print("2️⃣ Testing context-based generation...")
    context_docs = [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers."
    ]
    response = llm.generate_with_context(
        query="What is the difference between ML and deep learning?",
        retrieved_documents=context_docs
    )
    print(f"   Response: {response[:100]}...\n")
    
    # Test 3: Summarization
    print("3️⃣ Testing summarization...")
    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of intelligent agents: 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    """
    summary = llm.summarize_text(long_text, max_length=30)
    print(f"   Summary: {summary}\n")
    
    # Test 4: Concept explanation
    print("4️⃣ Testing concept explanation...")
    explanation = llm.explain_concept(
        concept="neural networks",
        audience="beginner"
    )
    print(f"   Explanation: {explanation[:100]}...\n")
    
    print("✅ All tests completed!")
    print("\n💡 Note: Some tests may show errors if GOOGLE_API_KEY is not set.")
    print("   Set your API key in .env file to enable full functionality.")
