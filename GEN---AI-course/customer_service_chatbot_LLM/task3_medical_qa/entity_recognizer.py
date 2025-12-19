"""
Medical Entity Recognizer - Task 3
Recognizes medical entities in text including symptoms, diseases, treatments, medications, body parts, and procedures.
Uses spaCy with medical NER models for entity extraction and classification.
"""

import re
from typing import List, Dict, Set
from dataclasses import dataclass


@dataclass
class MedicalEntity:
    """Represents a recognized medical entity"""
    text: str
    entity_type: str  # symptom, disease, treatment, medication, body_part, procedure
    confidence: float
    start_pos: int
    end_pos: int


class MedicalEntityRecognizer:
    """
    Recognizes and classifies medical entities in text.
    
    Uses pattern matching and keyword-based recognition for medical terminology.
    Can be extended with spaCy medical models (scispacy) for more advanced NER.
    """
    
    def __init__(self):
        """Initialize the medical entity recognizer with keyword dictionaries"""
        self._initialize_medical_keywords()
    
    def _initialize_medical_keywords(self):
        """Initialize dictionaries of medical keywords for each entity type"""
        
        # Common symptoms
        self.symptoms = {
            'pain', 'ache', 'fever', 'cough', 'fatigue', 'nausea', 'vomiting',
            'diarrhea', 'headache', 'dizziness', 'weakness', 'bleeding', 'swelling',
            'rash', 'itching', 'shortness of breath', 'chest pain', 'abdominal pain',
            'back pain', 'sore throat', 'runny nose', 'congestion', 'chills',
            'sweating', 'weight loss', 'weight gain', 'loss of appetite', 'insomnia',
            'anxiety', 'depression', 'confusion', 'numbness', 'tingling', 'tremor',
            'seizure', 'convulsion', 'paralysis', 'stiffness', 'inflammation',
            'discharge', 'bruising', 'cramps', 'spasms', 'palpitations', 'wheezing'
        }
        
        # Common diseases and conditions
        self.diseases = {
            'diabetes', 'cancer', 'hypertension', 'asthma', 'arthritis', 'pneumonia',
            'bronchitis', 'influenza', 'tuberculosis', 'malaria', 'hepatitis',
            'cirrhosis', 'stroke', 'heart disease', 'coronary artery disease',
            'heart attack', 'myocardial infarction', 'angina', 'arrhythmia',
            'leukemia', 'lymphoma', 'melanoma', 'carcinoma', 'sarcoma', 'tumor',
            'infection', 'sepsis', 'meningitis', 'encephalitis', 'alzheimer',
            'parkinson', 'multiple sclerosis', 'epilepsy', 'migraine', 'osteoporosis',
            'anemia', 'thrombosis', 'embolism', 'ulcer', 'gastritis', 'colitis',
            'nephritis', 'kidney disease', 'renal failure', 'liver disease',
            'thyroid disorder', 'hyperthyroidism', 'hypothyroidism', 'syndrome'
        }
        
        # Common treatments and procedures
        self.treatments = {
            'surgery', 'chemotherapy', 'radiation therapy', 'immunotherapy',
            'physical therapy', 'occupational therapy', 'dialysis', 'transplant',
            'transfusion', 'vaccination', 'immunization', 'injection', 'infusion',
            'therapy', 'treatment', 'procedure', 'operation', 'biopsy', 'endoscopy',
            'colonoscopy', 'angioplasty', 'bypass', 'resection', 'excision',
            'amputation', 'reconstruction', 'rehabilitation', 'counseling'
        }
        
        # Common medications
        self.medications = {
            'antibiotic', 'antibiotics', 'penicillin', 'amoxicillin', 'aspirin',
            'ibuprofen', 'acetaminophen', 'paracetamol', 'insulin', 'metformin',
            'statin', 'beta blocker', 'ace inhibitor', 'diuretic', 'antihistamine',
            'corticosteroid', 'steroid', 'prednisone', 'warfarin', 'heparin',
            'morphine', 'codeine', 'opioid', 'antidepressant', 'antipsychotic',
            'anticonvulsant', 'antiviral', 'antifungal', 'vaccine', 'medication',
            'drug', 'medicine', 'pill', 'tablet', 'capsule', 'syrup', 'ointment',
            'cream', 'lotion', 'drops', 'inhaler'
        }
        
        # Common body parts
        self.body_parts = {
            'heart', 'lung', 'lungs', 'liver', 'kidney', 'kidneys', 'brain',
            'stomach', 'intestine', 'colon', 'pancreas', 'spleen', 'bladder',
            'uterus', 'ovary', 'prostate', 'thyroid', 'bone', 'muscle', 'skin',
            'blood', 'artery', 'vein', 'nerve', 'spine', 'joint', 'eye', 'ear',
            'nose', 'throat', 'mouth', 'tongue', 'tooth', 'teeth', 'hand', 'foot',
            'arm', 'leg', 'chest', 'abdomen', 'back', 'neck', 'head', 'face'
        }
        
        # Common medical procedures
        self.procedures = {
            'x-ray', 'ct scan', 'mri', 'ultrasound', 'ecg', 'ekg', 'eeg',
            'blood test', 'urine test', 'biopsy', 'screening', 'examination',
            'checkup', 'diagnosis', 'scan', 'imaging', 'test', 'analysis'
        }
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[MedicalEntity]]:
        """
        Extract medical entities from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping entity types to lists of MedicalEntity objects
        """
        if not text:
            return {}
        
        text_lower = text.lower()
        entities = {
            'symptoms': [],
            'diseases': [],
            'treatments': [],
            'medications': [],
            'body_parts': [],
            'procedures': []
        }
        
        # Extract each entity type
        entities['symptoms'] = self._extract_entities(text, text_lower, self.symptoms, 'symptom')
        entities['diseases'] = self._extract_entities(text, text_lower, self.diseases, 'disease')
        entities['treatments'] = self._extract_entities(text, text_lower, self.treatments, 'treatment')
        entities['medications'] = self._extract_entities(text, text_lower, self.medications, 'medication')
        entities['body_parts'] = self._extract_entities(text, text_lower, self.body_parts, 'body_part')
        entities['procedures'] = self._extract_entities(text, text_lower, self.procedures, 'procedure')
        
        return entities
    
    def _extract_entities(self, original_text: str, text_lower: str, 
                         keywords: Set[str], entity_type: str) -> List[MedicalEntity]:
        """
        Extract entities of a specific type from text.
        
        Args:
            original_text: Original text with proper casing
            text_lower: Lowercase version of text
            keywords: Set of keywords to search for
            entity_type: Type of entity being extracted
            
        Returns:
            List of MedicalEntity objects found in text
        """
        entities = []
        
        for keyword in keywords:
            # Use word boundaries to match whole words
            pattern = r'\b' + re.escape(keyword) + r'\b'
            
            for match in re.finditer(pattern, text_lower):
                start_pos = match.start()
                end_pos = match.end()
                
                # Get the original text (with proper casing)
                matched_text = original_text[start_pos:end_pos]
                
                # Calculate confidence based on context (simple heuristic)
                confidence = self._calculate_confidence(text_lower, keyword, start_pos)
                
                entity = MedicalEntity(
                    text=matched_text,
                    entity_type=entity_type,
                    confidence=confidence,
                    start_pos=start_pos,
                    end_pos=end_pos
                )
                entities.append(entity)
        
        # Remove duplicates and sort by position
        entities = self._remove_overlapping_entities(entities)
        entities.sort(key=lambda e: e.start_pos)
        
        return entities
    
    def _calculate_confidence(self, text: str, keyword: str, position: int) -> float:
        """
        Calculate confidence score for an entity match.
        
        Args:
            text: Full text
            keyword: Matched keyword
            position: Position of match in text
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence
        confidence = 0.7
        
        # Increase confidence for longer keywords (more specific)
        if len(keyword) > 10:
            confidence += 0.1
        elif len(keyword) > 15:
            confidence += 0.15
        
        # Check for medical context words nearby
        context_window = 50
        start = max(0, position - context_window)
        end = min(len(text), position + len(keyword) + context_window)
        context = text[start:end]
        
        medical_context_words = ['patient', 'diagnosis', 'treatment', 'symptom', 
                                'condition', 'disease', 'medical', 'clinical']
        
        for word in medical_context_words:
            if word in context:
                confidence += 0.05
                break
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def _remove_overlapping_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """
        Remove overlapping entities, keeping the one with higher confidence.
        
        Args:
            entities: List of entities that may overlap
            
        Returns:
            List of non-overlapping entities
        """
        if not entities:
            return []
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda e: (e.start_pos, -e.confidence))
        
        result = []
        for entity in sorted_entities:
            # Check if this entity overlaps with any already added
            overlaps = False
            for existing in result:
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    overlaps = True
                    break
            
            if not overlaps:
                result.append(entity)
        
        return result
    
    def classify_entity_type(self, entity_text: str) -> str:
        """
        Classify a medical entity into its type.
        
        Args:
            entity_text: Text of the entity to classify
            
        Returns:
            Entity type (symptom, disease, treatment, medication, body_part, procedure)
        """
        entity_lower = entity_text.lower()
        
        if entity_lower in self.symptoms:
            return 'symptom'
        elif entity_lower in self.diseases:
            return 'disease'
        elif entity_lower in self.treatments:
            return 'treatment'
        elif entity_lower in self.medications:
            return 'medication'
        elif entity_lower in self.body_parts:
            return 'body_part'
        elif entity_lower in self.procedures:
            return 'procedure'
        else:
            return 'unknown'
    
    def get_entity_summary(self, entities: Dict[str, List[MedicalEntity]]) -> Dict[str, int]:
        """
        Get a summary count of entities by type.
        
        Args:
            entities: Dictionary of entities by type
            
        Returns:
            Dictionary with counts for each entity type
        """
        summary = {}
        for entity_type, entity_list in entities.items():
            summary[entity_type] = len(entity_list)
        return summary


if __name__ == "__main__":
    # Example usage
    recognizer = MedicalEntityRecognizer()
    
    # Test with sample medical text
    sample_text = """
    The patient presents with severe chest pain and shortness of breath. 
    Diagnosis suggests possible heart disease or pneumonia. 
    Treatment plan includes aspirin, beta blockers, and chest x-ray.
    The patient has a history of diabetes and hypertension.
    """
    
    print("Medical Entity Recognition Example")
    print("=" * 70)
    print(f"Text: {sample_text.strip()}")
    print("\n" + "=" * 70)
    
    entities = recognizer.extract_medical_entities(sample_text)
    
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"\n{entity_type.upper()}:")
            for entity in entity_list:
                print(f"  - {entity.text} (confidence: {entity.confidence:.2f})")
    
    # Summary
    summary = recognizer.get_entity_summary(entities)
    print("\n" + "=" * 70)
    print("Summary:")
    for entity_type, count in summary.items():
        if count > 0:
            print(f"  {entity_type}: {count}")
