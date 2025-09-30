import os
import json
from datetime import datetime
from pathlib import Path

def create_directories():
    """Create necessary directories for file storage"""
    directories = ['uploads', 'summaries', 'exports', 'src']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_metadata(result_data):
    """Save processing metadata to JSON file"""
    metadata_file = 'metadata.json'
    
    # Load existing metadata
    metadata = load_metadata()
    
    # Add new result
    metadata.append(result_data)
    
    # Save updated metadata
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def load_metadata():
    """Load processing metadata from JSON file"""
    metadata_file = 'metadata.json'
    
    if not os.path.exists(metadata_file):
        return []
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def clean_text(text):
    """Clean and preprocess extracted text"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove common PDF artifacts
    text = text.replace('\x00', '')  # null bytes
    text = text.replace('\uf0b7', '•')  # bullet points
    text = text.replace('\u2022', '•')  # bullet points
    
    # Fix common encoding issues
    text = text.replace(''', "'")
    text = text.replace(''', "'")
    text = text.replace('"', '"')
    text = text.replace('"', '"')
    text = text.replace('–', '-')
    text = text.replace('—', '-')
    
    return text.strip()

def calculate_optimal_summary_length(text_length, pages=1):
    """Calculate optimal summary length based on content length"""
    words_per_page = text_length / max(pages, 1)
    
    if words_per_page < 100:
        return 30, 80  # min, max
    elif words_per_page < 300:
        return 50, 120
    elif words_per_page < 600:
        return 80, 180
    else:
        return 100, 250

def format_timestamp(timestamp_str):
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str

def get_file_size_mb(file_path):
    """Get file size in MB"""
    try:
        return round(os.path.getsize(file_path) / (1024 * 1024), 2)
    except:
        return 0