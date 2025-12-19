"""
Utility Functions
Common helper functions used across all tasks.
Provides file operations, logging, error handling, and data processing utilities.
"""

import json
import os
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import hashlib
from pathlib import Path


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to save logs
        log_format: Optional custom log format
        
    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console output
        ]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"✅ Logging configured (level: {log_level})")
    
    return logger


# ============================================================================
# File Operations
# ============================================================================

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary containing JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"✅ Loaded JSON from: {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"❌ File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"❌ Invalid JSON in {file_path}: {e}")
        raise


def save_json(
    data: Dict[str, Any],
    file_path: str,
    indent: int = 2,
    ensure_dir: bool = True
) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save JSON file
        indent: Indentation level for pretty printing
        ensure_dir: Create directory if it doesn't exist
    """
    try:
        if ensure_dir:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        
        logging.info(f"✅ Saved JSON to: {file_path}")
    except Exception as e:
        logging.error(f"❌ Failed to save JSON to {file_path}: {e}")
        raise


def load_text(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Load text from a file.
    
    Args:
        file_path: Path to text file
        encoding: File encoding (default: utf-8)
        
    Returns:
        File contents as string
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            text = f.read()
        logging.info(f"✅ Loaded text from: {file_path} ({len(text)} chars)")
        return text
    except Exception as e:
        logging.error(f"❌ Failed to load text from {file_path}: {e}")
        raise


def save_text(
    text: str,
    file_path: str,
    encoding: str = 'utf-8',
    ensure_dir: bool = True
) -> None:
    """
    Save text to a file.
    
    Args:
        text: Text content to save
        file_path: Path to save file
        encoding: File encoding (default: utf-8)
        ensure_dir: Create directory if it doesn't exist
    """
    try:
        if ensure_dir:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(text)
        
        logging.info(f"✅ Saved text to: {file_path} ({len(text)} chars)")
    except Exception as e:
        logging.error(f"❌ Failed to save text to {file_path}: {e}")
        raise


def load_lines(file_path: str, encoding: str = 'utf-8') -> List[str]:
    """
    Load lines from a file.
    
    Args:
        file_path: Path to text file
        encoding: File encoding
        
    Returns:
        List of lines (without newline characters)
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            lines = [line.strip() for line in f.readlines()]
        logging.info(f"✅ Loaded {len(lines)} lines from: {file_path}")
        return lines
    except Exception as e:
        logging.error(f"❌ Failed to load lines from {file_path}: {e}")
        raise


def ensure_directory(directory: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)
    logging.debug(f"✅ Directory ensured: {directory}")


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path)


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    return get_file_size(file_path) / (1024 * 1024)


# ============================================================================
# Error Response Formatting
# ============================================================================

def create_error_response(
    error_code: str,
    error_message: str,
    user_message: str,
    retry_possible: bool = True,
    suggested_action: str = ""
) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        error_code: Error code identifier
        error_message: Technical error message
        user_message: User-friendly error message
        retry_possible: Whether the operation can be retried
        suggested_action: Suggested action for the user
        
    Returns:
        Dictionary with error information
    """
    return {
        'success': False,
        'error_code': error_code,
        'error_message': error_message,
        'user_message': user_message,
        'retry_possible': retry_possible,
        'suggested_action': suggested_action,
        'timestamp': datetime.now().isoformat()
    }


def create_success_response(
    data: Any,
    message: str = "Operation successful"
) -> Dict[str, Any]:
    """
    Create a standardized success response.
    
    Args:
        data: Response data
        message: Success message
        
    Returns:
        Dictionary with success information
    """
    return {
        'success': True,
        'data': data,
        'message': message,
        'timestamp': datetime.now().isoformat()
    }


# ============================================================================
# Data Processing Utilities
# ============================================================================

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Useful for processing long documents that exceed model token limits.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    logging.info(f"✅ Split text into {len(chunks)} chunks")
    return chunks


def truncate_text(
    text: str,
    max_length: int = 100,
    suffix: str = "..."
) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and special characters.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def generate_id(text: str) -> str:
    """
    Generate a unique ID from text using hash.
    
    Args:
        text: Text to generate ID from
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.md5(text.encode()).hexdigest()


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split items into batches.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


# ============================================================================
# Time and Date Utilities
# ============================================================================

def get_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        ISO formatted timestamp string
    """
    return datetime.now().isoformat()


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_file_exists(file_path: str) -> bool:
    """
    Check if a file exists.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file exists, False otherwise
    """
    exists = os.path.isfile(file_path)
    if not exists:
        logging.warning(f"⚠️ File not found: {file_path}")
    return exists


def validate_directory_exists(directory: str) -> bool:
    """
    Check if a directory exists.
    
    Args:
        directory: Path to directory
        
    Returns:
        True if directory exists, False otherwise
    """
    exists = os.path.isdir(directory)
    if not exists:
        logging.warning(f"⚠️ Directory not found: {directory}")
    return exists


def validate_not_empty(text: str) -> bool:
    """
    Check if text is not empty.
    
    Args:
        text: Text to validate
        
    Returns:
        True if text is not empty, False otherwise
    """
    return bool(text and text.strip())


# ============================================================================
# Progress Tracking
# ============================================================================

class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            description: Description of the operation
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1) -> None:
        """
        Update progress.
        
        Args:
            increment: Number of items completed
        """
        self.current += increment
        percentage = (self.current / self.total) * 100
        
        if self.current % max(1, self.total // 10) == 0 or self.current == self.total:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            logging.info(
                f"📊 {self.description}: {self.current}/{self.total} "
                f"({percentage:.1f}%) - {format_duration(elapsed)}"
            )
    
    def complete(self) -> None:
        """Mark progress as complete."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logging.info(
            f"✅ {self.description} complete! "
            f"Total time: {format_duration(elapsed)}"
        )


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing Utility Functions\n")
    
    # Set up logging
    logger = setup_logging(log_level="INFO")
    
    # Test file operations
    print("1️⃣ Testing file operations...")
    test_data = {"name": "Test", "value": 123, "items": [1, 2, 3]}
    save_json(test_data, "./test_output/test.json")
    loaded_data = load_json("./test_output/test.json")
    print(f"   Loaded data: {loaded_data}\n")
    
    # Test text operations
    print("2️⃣ Testing text operations...")
    test_text = "This is a test text with multiple    spaces"
    cleaned = clean_text(test_text)
    print(f"   Original: '{test_text}'")
    print(f"   Cleaned: '{cleaned}'\n")
    
    # Test chunking
    print("3️⃣ Testing text chunking...")
    long_text = "A" * 500
    chunks = chunk_text(long_text, chunk_size=100, overlap=20)
    print(f"   Split {len(long_text)} chars into {len(chunks)} chunks\n")
    
    # Test error response
    print("4️⃣ Testing error response...")
    error = create_error_response(
        error_code="TEST_ERROR",
        error_message="This is a test error",
        user_message="Something went wrong",
        suggested_action="Try again later"
    )
    print(f"   Error response: {error}\n")
    
    # Test progress tracker
    print("5️⃣ Testing progress tracker...")
    tracker = ProgressTracker(total=10, description="Test operation")
    for i in range(10):
        tracker.update()
    tracker.complete()
    
    print("\n✅ All utility tests passed!")
