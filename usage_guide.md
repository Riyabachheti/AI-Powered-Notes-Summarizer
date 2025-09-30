# 🚀 Quick Start Guide - AI-Powered Notes Summarizer

## 🏃‍♂️ Run the Application

### Method 1: Direct Command
```bash
cd /app
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

### Method 2: Using Startup Script
```bash
cd /app
./start_app.sh
```

### Method 3: Background Process
```bash
cd /app
nohup streamlit run app.py --server.port=8501 --server.address=0.0.0.0 > app.log 2>&1 &
```

## 🌐 Access the Application
- **URL**: http://localhost:8501
- **Interface**: Modern Streamlit web application
- **Browser**: Works in any modern browser

## ⚡ Quick Test

### Test with Sample File
1. The app includes `/app/sample_text.txt` for testing
2. Upload this file or paste its content
3. Click "🚀 Generate Summary"
4. Download results as PDF or TXT

### Test Command Line
```bash
cd /app
python -c "
from src.text_summarizer import TextSummarizer
from src.keyword_extractor import KeywordExtractor

# Quick test
text = 'Artificial intelligence is transforming healthcare through advanced diagnostic tools and personalized medicine solutions.'
summarizer = TextSummarizer()
summary = summarizer.summarize(text, model_name='t5-small')
extractor = KeywordExtractor()
keywords = extractor.extract_keywords(text, 5)
print(f'Summary: {summary}')
print(f'Keywords: {keywords}')
"
```

## 📁 Project Structure
```
/app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
├── start_app.sh          # Startup script
├── src/                  # Core modules
│   ├── text_summarizer.py
│   ├── keyword_extractor.py
│   ├── file_processor.py
│   └── export_manager.py
├── uploads/              # File uploads
├── exports/              # Generated exports
└── sample_text.txt       # Test file
```

## 🎯 Key Features
- **Multiple AI Models**: BART, T5, Pegasus
- **File Support**: PDF, DOCX, TXT
- **Smart Keywords**: Semantic extraction
- **Professional Exports**: PDF/TXT with metadata
- **Processing History**: Track all summaries

## 🔧 Troubleshooting

### Application Won't Start
```bash
# Kill any existing processes
pkill -f streamlit

# Check dependencies
pip install -r requirements.txt

# Restart application
cd /app && streamlit run app.py --server.port=8501
```

### Models Not Loading
- First run takes longer (downloading models)
- Ensure stable internet connection
- Check available disk space (models ~1-2GB)

### Port Already in Use
```bash
# Use different port
streamlit run app.py --server.port=8502
```

## 🎊 You're Ready!
The AI-Powered Notes Summarizer is now ready to transform your document processing workflow!