"""
Task 6: Multi-Lingual Support System
Implements automatic language detection, translation, and culturally appropriate responses.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import re
from dataclasses import dataclass, field
from datetime import datetime

# Language detection and translation
from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException

# Translation models
from transformers import pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LanguageResult:
    """Data class for language detection results."""
    language: str  # ISO 639-1 language code
    confidence: float  # confidence score 0-1
    alternatives: List[Dict[str, float]] = field(default_factory=list)

@dataclass
class TranslationResult:
    """Data class for translation results."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float = 0.0

@dataclass
class MultiLingualContext:
    """Context for multi-lingual conversations."""
    conversation_id: str
    primary_language: str
    language_history: List[str] = field(default_factory=list)
    translation_cache: Dict[str, str] = field(default_factory=dict)
    cultural_preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

class MultiLingualSystem:
    """
    Multi-lingual support system for automatic language detection, translation,
    and culturally appropriate response generation.
    
    Supports: English (en), Hindi (hi), Spanish (es), French (fr)
    """
    
    def __init__(self, supported_languages: List[str] = None):
        """Initialize the multi-lingual system."""
        # Default supported languages
        self.supported_languages = supported_languages or ['en', 'hi', 'es', 'fr']
        
        # Language mappings
        self.language_names = {
            'en': 'English',
            'hi': 'Hindi', 
            'es': 'Spanish',
            'fr': 'French'
        }
        
        # Translation models cache
        self.translation_pipelines = {}
        
        # Conversation contexts
        self.conversation_contexts: Dict[str, MultiLingualContext] = {}
        
        # Cultural adaptation rules
        self.cultural_rules = self._initialize_cultural_rules()
        
        # Initialize translation models
        self._initialize_translation_models()
        
        logger.info(f"✅ Multi-lingual system initialized with languages: {self.supported_languages}")
    
    def _initialize_cultural_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cultural adaptation rules for each language."""
        return {
            'en': {
                'greetings': ['Hello', 'Hi', 'Good day', 'Welcome'],
                'closings': ['Thank you', 'Best regards', 'Have a great day'],
                'politeness_level': 'moderate',
                'formality': 'professional'
            },
            'hi': {
                'greetings': ['नमस्ते', 'नमस्कार', 'आपका स्वागत है'],
                'closings': ['धन्यवाद', 'आपका दिन शुभ हो', 'सादर'],
                'politeness_level': 'high',
                'formality': 'respectful'
            },
            'es': {
                'greetings': ['Hola', 'Buenos días', 'Bienvenido', 'Saludos'],
                'closings': ['Gracias', 'Que tenga un buen día', 'Saludos cordiales'],
                'politeness_level': 'high',
                'formality': 'warm'
            },
            'fr': {
                'greetings': ['Bonjour', 'Salut', 'Bienvenue', 'Bonsoir'],
                'closings': ['Merci', 'Bonne journée', 'Cordialement'],
                'politeness_level': 'high',
                'formality': 'elegant'
            }
        }
    
    def _initialize_translation_models(self):
        """Initialize translation models for supported language pairs."""
        try:
            # Helsinki-NLP models for translation
            model_pairs = [
                ('en', 'es'),  # English to Spanish
                ('es', 'en'),  # Spanish to English
                ('en', 'fr'),  # English to French
                ('fr', 'en'),  # French to English
            ]
            
            for src, tgt in model_pairs:
                model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
                try:
                    pipeline_obj = pipeline(
                        "translation",
                        model=model_name,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    self.translation_pipelines[f"{src}-{tgt}"] = pipeline_obj
                    logger.info(f"✅ Loaded translation model: {src} -> {tgt}")
                except Exception as e:
                    logger.warning(f"Failed to load translation model {src}->{tgt}: {e}")
                    
        except Exception as e:
            logger.error(f"Error initializing translation models: {e}")
    
    def detect_language(self, text: str) -> LanguageResult:
        """Detect the language of input text."""
        if not text or not text.strip():
            return LanguageResult(language='en', confidence=0.5, alternatives=[])
        
        try:
            # Clean text for better detection
            cleaned_text = re.sub(r'[^\w\s]', ' ', text.strip())
            
            if len(cleaned_text) < 3:
                return LanguageResult(language='en', confidence=0.5, alternatives=[])
            
            # Detect language with alternatives
            detected_langs = detect_langs(cleaned_text)
            
            # Get primary language
            primary_lang = detected_langs[0]
            
            # Format alternatives
            alternatives = []
            for lang_obj in detected_langs[1:5]:  # Top 5 alternatives
                alternatives.append({
                    'language': lang_obj.lang,
                    'confidence': lang_obj.prob
                })
            
            # Check if detected language is supported
            detected_code = primary_lang.lang
            if detected_code not in self.supported_languages:
                detected_code = 'en'  # Default to English
                confidence = 0.6
            else:
                confidence = primary_lang.prob
            
            return LanguageResult(
                language=detected_code,
                confidence=confidence,
                alternatives=alternatives
            )
            
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}")
            return LanguageResult(language='en', confidence=0.5, alternatives=[])
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            return LanguageResult(language='en', confidence=0.5, alternatives=[])
    
    def translate_text(self, text: str, target_lang: str, source_lang: str = "auto") -> TranslationResult:
        """Translate text from source language to target language."""
        if not text or not text.strip():
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language='en',
                target_language=target_lang
            )
        
        # Auto-detect source language if needed
        if source_lang == "auto":
            detection_result = self.detect_language(text)
            source_lang = detection_result.language
        
        # If source and target are the same, return original
        if source_lang == target_lang:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_lang,
                target_language=target_lang,
                confidence=1.0
            )
        
        try:
            # Get translation pipeline
            pipeline_key = f"{source_lang}-{target_lang}"
            
            if pipeline_key in self.translation_pipelines:
                pipeline_obj = self.translation_pipelines[pipeline_key]
                result = pipeline_obj(text)
                translated_text = result[0]['translation_text']
                confidence = 0.8
            else:
                # No direct translation available
                translated_text = text
                confidence = 0.0
                logger.warning(f"No translation model available for {source_lang} -> {target_lang}")
            
            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=source_lang,
                target_language=target_lang,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Translation failed for {source_lang} -> {target_lang}: {e}")
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_lang,
                target_language=target_lang,
                confidence=0.0
            )
    
    def process_multilingual_query(self, query: str, conversation_id: str = None) -> Dict[str, Any]:
        """Process a multi-lingual query with language detection and context management."""
        # Generate conversation ID if not provided
        if conversation_id is None:
            conversation_id = f"ml_conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Detect language
        language_result = self.detect_language(query)
        detected_lang = language_result.language
        
        # Get or create conversation context
        context = self._get_or_create_context(conversation_id, detected_lang)
        
        # Update language history
        context.language_history.append(detected_lang)
        if len(context.language_history) > 10:
            context.language_history = context.language_history[-10:]
        
        # Translate to English if needed
        english_query = query
        translation_result = None
        
        if detected_lang != 'en':
            translation_result = self.translate_text(query, 'en', detected_lang)
            english_query = translation_result.translated_text
        
        return {
            'original_query': query,
            'english_query': english_query,
            'detected_language': detected_lang,
            'language_confidence': language_result.confidence,
            'translation_result': translation_result,
            'conversation_id': conversation_id,
            'context': context
        }
    
    def generate_culturally_appropriate_response(self, response: str, target_lang: str) -> str:
        """Generate culturally appropriate response for target language."""
        if target_lang not in self.supported_languages:
            return response
        
        try:
            # Translate response if needed
            if target_lang != 'en':
                translation_result = self.translate_text(response, target_lang, 'en')
                adapted_response = translation_result.translated_text
            else:
                adapted_response = response
            
            # Apply cultural adaptations
            cultural_rules = self.cultural_rules.get(target_lang, self.cultural_rules['en'])
            adapted_response = self._apply_cultural_adaptations(adapted_response, target_lang, cultural_rules)
            
            return adapted_response
            
        except Exception as e:
            logger.error(f"Error generating culturally appropriate response: {e}")
            return response
    
    def _apply_cultural_adaptations(self, response: str, target_lang: str, cultural_rules: Dict[str, Any]) -> str:
        """Apply cultural adaptations to response."""
        adapted_response = response
        
        # Apply language-specific cultural adaptations
        if target_lang == 'hi':  # Hindi - respectful and formal
            adapted_response = self._apply_hindi_cultural_rules(adapted_response, cultural_rules)
        elif target_lang == 'es':  # Spanish - warm and personal
            adapted_response = self._apply_spanish_cultural_rules(adapted_response, cultural_rules)
        elif target_lang == 'fr':  # French - formal and elegant
            adapted_response = self._apply_french_cultural_rules(adapted_response, cultural_rules)
        else:  # English - professional and direct
            adapted_response = self._apply_english_cultural_rules(adapted_response, cultural_rules)
        
        return adapted_response
    
    def _apply_hindi_cultural_rules(self, response: str, cultural_rules: Dict[str, Any]) -> str:
        """Apply Hindi cultural adaptations - respectful and formal."""
        # Add respectful greeting if not present
        if not any(greeting in response for greeting in cultural_rules['greetings']):
            if len(response.split()) < 8:  # Short responses get greeting
                greeting = cultural_rules['greetings'][0]  # नमस्ते
                response = f"{greeting}! {response}"
        
        # Add respectful closing for longer responses
        if len(response.split()) > 15:
            if not any(closing in response for closing in cultural_rules['closings']):
                closing = cultural_rules['closings'][0]  # धन्यवाद
                response = f"{response} {closing}।"
        
        return response
    
    def _apply_spanish_cultural_rules(self, response: str, cultural_rules: Dict[str, Any]) -> str:
        """Apply Spanish cultural adaptations - warm and personal."""
        # Add warm greeting
        if not any(greeting.lower() in response.lower() for greeting in cultural_rules['greetings']):
            if 'help' in response.lower() or 'assist' in response.lower():
                greeting = cultural_rules['greetings'][0]  # Hola
                response = f"¡{greeting}! {response}"
        
        # Make responses warmer by adding personal touches
        if 'thank' in response.lower():
            response = response.replace('Thank you', 'Muchas gracias')
            response = response.replace('thanks', 'gracias')
        
        # Add warm closing for service responses
        if 'service' in response.lower() or 'help' in response.lower():
            if not any(closing in response for closing in cultural_rules['closings']):
                response = f"{response} ¡Que tenga un excelente día!"
        
        return response
    
    def _apply_french_cultural_rules(self, response: str, cultural_rules: Dict[str, Any]) -> str:
        """Apply French cultural adaptations - formal and elegant."""
        # Add formal greeting
        if not any(greeting.lower() in response.lower() for greeting in cultural_rules['greetings']):
            if len(response.split()) < 10:
                greeting = cultural_rules['greetings'][0]  # Bonjour
                response = f"{greeting}! {response}"
        
        # Make responses more formal and elegant
        response = response.replace('Hi', 'Bonjour')
        response = response.replace('Hello', 'Bonjour')
        response = response.replace('Thanks', 'Merci beaucoup')
        
        # Add elegant closing
        if len(response.split()) > 12:
            if not any(closing in response for closing in cultural_rules['closings']):
                closing = cultural_rules['closings'][2]  # Cordialement
                response = f"{response} {closing}."
        
        return response
    
    def _apply_english_cultural_rules(self, response: str, cultural_rules: Dict[str, Any]) -> str:
        """Apply English cultural adaptations - professional and direct."""
        # Add professional greeting for service contexts
        if not any(greeting.lower() in response.lower() for greeting in cultural_rules['greetings']):
            if 'service' in response.lower() or 'support' in response.lower():
                greeting = cultural_rules['greetings'][0]  # Hello
                response = f"{greeting}! {response}"
        
        # Ensure professional tone
        response = response.replace('hey', 'hello')
        response = response.replace('yeah', 'yes')
        
        return response
    
    def _get_or_create_context(self, conversation_id: str, primary_language: str) -> MultiLingualContext:
        """Get or create conversation context."""
        if conversation_id not in self.conversation_contexts:
            self.conversation_contexts[conversation_id] = MultiLingualContext(
                conversation_id=conversation_id,
                primary_language=primary_language
            )
        return self.conversation_contexts[conversation_id]
    
    def maintain_multilingual_context(self, conversation_id: str, language: str):
        """Maintain multi-lingual context for conversation."""
        context = self._get_or_create_context(conversation_id, language)
        
        # Update primary language if this language is used frequently
        if context.language_history.count(language) > len(context.language_history) // 2:
            context.primary_language = language
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages with their names."""
        return {code: self.language_names[code] for code in self.supported_languages}
    
    def get_cultural_greeting(self, language: str, context: str = "general") -> str:
        """Get culturally appropriate greeting for a language."""
        if language not in self.cultural_rules:
            language = 'en'
        
        greetings = self.cultural_rules[language]['greetings']
        
        # Context-specific greetings
        if context == "formal":
            return greetings[1] if len(greetings) > 1 else greetings[0]
        elif context == "casual":
            return greetings[-1] if len(greetings) > 1 else greetings[0]
        else:
            return greetings[0]  # Default greeting
    
    def get_cultural_closing(self, language: str, context: str = "general") -> str:
        """Get culturally appropriate closing for a language."""
        if language not in self.cultural_rules:
            language = 'en'
        
        closings = self.cultural_rules[language]['closings']
        
        # Context-specific closings
        if context == "formal":
            return closings[-1] if len(closings) > 1 else closings[0]
        elif context == "casual":
            return closings[0]
        else:
            return closings[1] if len(closings) > 1 else closings[0]
    
    def adapt_response_formality(self, response: str, language: str, formality_level: str = "auto") -> str:
        """Adapt response formality based on cultural expectations."""
        if language not in self.cultural_rules:
            return response
        
        cultural_rules = self.cultural_rules[language]
        expected_formality = cultural_rules.get('formality', 'professional')
        
        # Auto-detect formality level if not specified
        if formality_level == "auto":
            formality_level = expected_formality
        
        adapted_response = response
        
        if formality_level == "respectful" and language == "hi":
            # Make Hindi responses more respectful
            adapted_response = adapted_response.replace("you", "आप")
            adapted_response = adapted_response.replace("your", "आपका")
            
        elif formality_level == "warm" and language == "es":
            # Make Spanish responses warmer
            adapted_response = adapted_response.replace("Hello", "¡Hola")
            adapted_response = adapted_response.replace("Good", "¡Muy bien")
            
        elif formality_level == "elegant" and language == "fr":
            # Make French responses more elegant
            adapted_response = adapted_response.replace("Hello", "Bonjour")
            adapted_response = adapted_response.replace("Good", "Très bien")
        
        return adapted_response
    
    def get_language_specific_phrases(self, language: str) -> Dict[str, List[str]]:
        """Get language-specific phrases for common interactions."""
        phrases = {
            'en': {
                'acknowledgment': ['I understand', 'I see', 'Got it'],
                'apology': ['I apologize', 'Sorry about that', 'My apologies'],
                'assistance': ['How can I help?', 'What can I do for you?', 'I\'m here to assist'],
                'confirmation': ['Certainly', 'Of course', 'Absolutely'],
                'patience': ['Please wait', 'One moment please', 'Just a moment']
            },
            'hi': {
                'acknowledgment': ['मैं समझ गया', 'मुझे पता है', 'ठीक है'],
                'apology': ['मुझे खुशी है', 'माफ करें', 'क्षमा करें'],
                'assistance': ['मैं कैसे मदद कर सकता हूं?', 'आपकी क्या सेवा कर सकता हूं?', 'मैं यहां सहायता के लिए हूं'],
                'confirmation': ['निश्चित रूप से', 'बिल्कुल', 'जरूर'],
                'patience': ['कृपया प्रतीक्षा करें', 'एक क्षण रुकें', 'थोड़ा इंतजार करें']
            },
            'es': {
                'acknowledgment': ['Entiendo', 'Ya veo', 'Comprendo'],
                'apology': ['Lo siento', 'Disculpe', 'Mis disculpas'],
                'assistance': ['¿Cómo puedo ayudarle?', '¿En qué puedo servirle?', 'Estoy aquí para ayudar'],
                'confirmation': ['Por supuesto', 'Claro que sí', 'Absolutamente'],
                'patience': ['Por favor espere', 'Un momento por favor', 'Solo un momento']
            },
            'fr': {
                'acknowledgment': ['Je comprends', 'Je vois', 'D\'accord'],
                'apology': ['Je suis désolé', 'Excusez-moi', 'Mes excuses'],
                'assistance': ['Comment puis-je vous aider?', 'Que puis-je faire pour vous?', 'Je suis là pour vous assister'],
                'confirmation': ['Certainement', 'Bien sûr', 'Absolument'],
                'patience': ['Veuillez patienter', 'Un moment s\'il vous plaît', 'Juste un moment']
            }
        }
        
        return phrases.get(language, phrases['en'])
    
    def enhance_response_with_cultural_phrases(self, response: str, language: str, context: str = "general") -> str:
        """Enhance response with culturally appropriate phrases."""
        if language not in self.supported_languages:
            return response
        
        phrases = self.get_language_specific_phrases(language)
        enhanced_response = response
        
        # Add acknowledgment for questions
        if '?' in response or 'question' in response.lower():
            if not any(ack.lower() in response.lower() for ack in phrases['acknowledgment']):
                acknowledgment = phrases['acknowledgment'][0]
                enhanced_response = f"{acknowledgment}. {enhanced_response}"
        
        # Add confirmation for positive responses
        if any(word in response.lower() for word in ['yes', 'correct', 'right', 'exactly']):
            if not any(conf.lower() in response.lower() for conf in phrases['confirmation']):
                confirmation = phrases['confirmation'][0]
                enhanced_response = f"{confirmation}! {enhanced_response}"
        
        # Add patience request for processing responses
        if any(word in response.lower() for word in ['processing', 'checking', 'looking']):
            patience = phrases['patience'][0]
            enhanced_response = f"{patience}. {enhanced_response}"
        
        return enhanced_response

# Example usage
if __name__ == "__main__":
    ml_system = MultiLingualSystem()
    
    # Test language detection
    test_texts = [
        "Hello, how are you today?",
        "Hola, ¿cómo estás hoy?",
        "Bonjour, comment allez-vous?"
    ]
    
    print("Language Detection Test:")
    for text in test_texts:
        result = ml_system.detect_language(text)
        print(f"'{text}' -> {result.language} (confidence: {result.confidence:.3f})")
    
    # Test translation
    print("\nTranslation Test:")
    english_text = "Hello, how can I help you?"
    for lang in ['es', 'fr']:
        translation = ml_system.translate_text(english_text, lang, 'en')
        print(f"EN->'{lang}': {translation.translated_text}")