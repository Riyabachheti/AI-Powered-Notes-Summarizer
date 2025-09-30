#!/usr/bin/env python3
"""
AI-Powered Notes Summarizer Runner
Streamlit application startup script
"""

import sys
import os
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import streamlit
        import torch
        import transformers
        logger.info("Core dependencies verified")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def install_spacy_model():
    """Install spaCy English model if not present"""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy English model already installed")
        except OSError:
            logger.info("Installing spaCy English model...")
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                         check=True, capture_output=True)
            logger.info("spaCy English model installed successfully")
    except Exception as e:
        logger.warning(f"Could not install spaCy model: {e}")

def setup_environment():
    """Setup the environment for the application"""
    # Create necessary directories
    directories = ['uploads', 'summaries', 'exports', 'src']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created/verified directory: {directory}")
    
    # Install spaCy model
    install_spacy_model()
    
    logger.info("Environment setup completed")

def main():
    """Main runner function"""
    print("🚀 Starting AI-Powered Notes Summarizer...")
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Missing dependencies. Please install requirements.txt")
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Run Streamlit app
    try:
        logger.info("Starting Streamlit application...")
        os.system("streamlit run app.py --server.port=8501 --server.address=0.0.0.0")
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        logger.error(f"Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()