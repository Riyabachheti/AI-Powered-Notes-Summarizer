# 🤖 AI-Powered Notes Summarizer

A comprehensive **Streamlit-based application** that leverages **HuggingFace Transformers** to intelligently summarize documents and extract key insights from various text formats.

## 🎯 Features

### 📄 Multi-Format Support
- **PDF Files**: Extract text from multi-page PDFs using PyPDF2
- **Word Documents**: Process .docx files with python-docx  
- **Text Files**: Handle plain text files (.txt)
- **Direct Input**: Paste raw text directly into the interface

### 🧠 AI-Powered Summarization
- **Multiple Models**: Choose from facebook/bart-large-cnn, t5-small, google/pegasus-xsum
- **Smart Length Adaptation**: Medium length per page (automatic sizing based on content)
- **Chunk Processing**: Handles long documents by intelligently chunking text
- **Model-Specific Optimization**: Each model optimized for different content types

### 🔑 Intelligent Keyword Extraction
- **KeyBERT**: Advanced semantic keyword extraction
- **spaCy NER**: Named entity recognition for people, organizations, locations
- **RAKE Algorithm**: Key phrase extraction for better context
- **NLTK Fallback**: Robust keyword extraction with POS tagging

### 💾 Export & History
- **PDF Export**: Professional summary reports with metadata
- **TXT Export**: Plain text summaries for easy sharing
- **Processing History**: Track all your summarization tasks
- **Metadata Tracking**: File info, timestamps, model used, compression ratios

## 🚀 Quick Start

### Installation
```bash
# Clone or navigate to the project directory
cd /app

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm

# Create necessary directories
mkdir -p uploads summaries exports
```

### Run the Application
```bash
# Option 1: Direct Streamlit
streamlit run app.py --server.port=8501 --server.address=0.0.0.0

# Option 2: Using startup script
./start_app.sh

# Option 3: Using Python runner
python run_app.py
```

The application will be available at: **http://localhost:8501**

## 🛠️ Technology Stack

### Core Technologies
- **Frontend**: Streamlit (Interactive web interface)
- **AI/ML**: HuggingFace Transformers (BART, T5, Pegasus)
- **NLP**: KeyBERT, spaCy, NLTK
- **File Processing**: PyPDF2, python-docx
- **Export**: FPDF2, ReportLab

### AI Models
- **facebook/bart-large-cnn**: Optimized for news and article summarization
- **t5-small**: Versatile text-to-text transformer (faster inference)  
- **google/pegasus-xsum**: Specialized for abstractive summarization

## 📖 How to Use

### 1. Choose Configuration
- Select your preferred summarization model
- Choose between automatic or custom summary length
- Enable/disable keyword extraction (5-20 keywords)

### 2. Input Your Content
- **Upload Files**: Drag & drop PDF, TXT, or DOCX files
- **Paste Text**: Copy and paste text directly into the interface

### 3. Generate Summary
- Click "🚀 Generate Summary" 
- Wait for processing (models download automatically on first use)
- View results with statistics and insights

### 4. Export Results
- Download summary as PDF with professional formatting
- Download as TXT for easy sharing
- View processing history and statistics

## 📊 Model Comparison

| Model | Best For | Speed | Quality | Max Input |
|-------|----------|--------|---------|-----------|
| BART-Large-CNN | News, Articles | Medium | Excellent | 1024 tokens |
| T5-Small | General Text | Fast | Good | 512 tokens |
| Pegasus-XSUM | Creative Summaries | Medium | Excellent | 512 tokens |

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── run_app.py            # Application runner with setup
├── start_app.sh          # Bash startup script
├── src/                  # Core modules
│   ├── __init__.py
│   ├── file_processor.py # File upload and text extraction
│   ├── text_summarizer.py # AI summarization engine
│   ├── keyword_extractor.py # Keyword extraction engine
│   ├── export_manager.py # Export to PDF/TXT
│   └── utils.py          # Helper functions
├── uploads/              # Uploaded files storage
├── summaries/            # Generated summaries
├── exports/              # Export files (PDF/TXT)
└── metadata.json         # Processing history
```

## 🔍 Advanced Features

### Smart Text Processing
- **Multi-page PDF handling** with page markers
- **Encoding detection** for various text formats
- **Text cleaning** removes PDF artifacts and fixes encoding
- **Chunk processing** for documents exceeding model limits

### Intelligent Summarization
- **Context preservation** when chunking long documents
- **Model-specific optimization** for different content types
- **Length adaptation** based on source document characteristics
- **Quality validation** with fallback mechanisms

### Comprehensive Analytics
- **Compression ratios** showing summarization efficiency
- **Word count statistics** for original vs. summary
- **Processing time tracking** for performance monitoring
- **Model performance comparison** across different content

## 🚨 Troubleshooting

### Common Issues

**Models not downloading:**
- Ensure stable internet connection
- First run may take longer for model downloads
- Check available disk space (models can be 1-2GB each)

**Memory issues:**
- Use T5-small for faster processing on limited hardware
- Process smaller documents or chunks
- Restart application if memory usage is high

**File upload errors:**
- Ensure file formats are supported (.pdf, .txt, .docx)
- Check file size limits (recommended < 10MB)
- Verify file is not corrupted or password-protected

### Performance Optimization
- **GPU Support**: Automatic CUDA detection for faster processing
- **Model Caching**: Models are cached after first load
- **Concurrent Processing**: Handle multiple files efficiently

## 🎨 User Interface

### Main Features
- **Clean, Professional Design** with Streamlit's modern UI
- **Real-time Processing Feedback** with progress indicators
- **Interactive Configuration** with helpful tooltips
- **Comprehensive Results Display** with statistics and insights

### Export Features
- **Professional PDF Reports** with metadata and formatting
- **Plain Text Exports** for easy integration
- **Processing History** with searchable records
- **Download Management** with organized file structure

## 🔒 Data Privacy

- **Local Processing**: All AI processing happens locally
- **No External APIs**: Models run on your machine
- **File Security**: Uploaded files stored locally only
- **Data Control**: Full control over your documents and summaries

## 📈 Performance Metrics

### Typical Processing Times
- **Small files** (< 1000 words): 5-15 seconds
- **Medium files** (1000-5000 words): 15-45 seconds
- **Large files** (> 5000 words): 45-120 seconds

*Times vary based on hardware and selected model*

## 🤝 Contributing

This is an application showcasing:
- **Advanced NLP Integration** with HuggingFace models
- **Professional UI/UX** with Streamlit
- **Robust File Processing** for multiple formats
- **Export and History Management**
- **Error Handling and Validation**
