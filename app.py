import streamlit as st
import os
import json
import uuid
from datetime import datetime
from pathlib import Path

# Import our custom modules
from src.file_processor import FileProcessor
from src.text_summarizer import TextSummarizer
from src.keyword_extractor import KeywordExtractor
from src.export_manager import ExportManager
from src.utils import create_directories, save_metadata, load_metadata

# Initialize directories
create_directories()

# Initialize components
@st.cache_resource
def load_models():
    """Load and cache ML models"""
    summarizer = TextSummarizer()
    keyword_extractor = KeywordExtractor()
    return summarizer, keyword_extractor

def main():
    st.set_page_config(
        page_title="AI-Powered Notes Summarizer",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🤖 AI-Powered Notes Summarizer")
    st.markdown("**Upload documents or paste text to generate intelligent summaries and extract key insights**")

    # Sidebar for model selection and settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "Choose Summarization Model:",
            ["facebook/bart-large-cnn", "t5-small", "google/pegasus-xsum"],
            help="BART: Best for news-style content\nT5: Versatile for various text types\nPegasus: Optimized for abstractive summaries"
        )
        
        # Summary settings
        st.subheader("Summary Settings")
        summary_style = st.radio(
            "Summary Style:",
            ["Automatic (per page)", "Custom length"],
            help="Automatic adapts length based on content"
        )
        
        if summary_style == "Custom length":
            max_length = st.slider("Max Summary Length (words):", 100, 800, 400)
            min_length = st.slider("Min Summary Length (words):", 50, 300, 100)
        else:
            max_length, min_length = None, None
       
        # Keyword extraction settings
        st.subheader("Keyword Extraction")
        extract_keywords = st.checkbox("Extract Keywords", value=True)
        num_keywords = st.slider("Number of Keywords:", 5, 20, 10) if extract_keywords else 0

    # Main content area
    tab1, tab2, tab3 = st.tabs(["📄 Process Documents", "📊 View History", "ℹ️ About"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Method")
            input_method = st.radio("Choose input method:", ["Upload Files", "Paste Text"])
            
            if input_method == "Upload Files":
                uploaded_files = st.file_uploader(
                    "Upload your documents:",
                    type=['pdf', 'txt', 'docx'],
                    accept_multiple_files=True,
                    help="Supported formats: PDF, TXT, DOCX"
                )
            else:
                uploaded_files = None
                raw_text = st.text_area(
                    "Paste your text here:",
                    height=200,
                    placeholder="Enter the text you want to summarize..."
                )
        
        with col2:
            st.subheader("Processing Options")
            
            # Initialize session state
            if 'processing_complete' not in st.session_state:
                st.session_state.processing_complete = False
            if 'results' not in st.session_state:
                st.session_state.results = None
            
            # Process button
            if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
                if (input_method == "Upload Files" and uploaded_files) or (input_method == "Paste Text" and raw_text.strip()):
                    with st.spinner("Processing... This may take a moment"):
                        try:
                            # Load models
                            summarizer, keyword_extractor = load_models()
                            
                            # Initialize processors
                            file_processor = FileProcessor()
                            
                            results = []
                            
                            if input_method == "Upload Files":
                                for uploaded_file in uploaded_files:
                                    result = process_file(
                                        uploaded_file, file_processor, summarizer, 
                                        keyword_extractor, model_choice, extract_keywords, 
                                        num_keywords, max_length, min_length
                                    )
                                    if result:
                                        results.append(result)
                            else:
                                result = process_text(
                                    raw_text, summarizer, keyword_extractor, 
                                    model_choice, extract_keywords, num_keywords, 
                                    max_length, min_length
                                )
                                if result:
                                    results.append(result)
                            
                            st.session_state.results = results
                            st.session_state.processing_complete = True
                            st.success("✅ Processing completed successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Error during processing: {str(e)}")
                            st.session_state.processing_complete = False
                else:
                    st.warning("⚠️ Please upload files or paste text to process.")
        
        # Display results
        if st.session_state.processing_complete and st.session_state.results:
            display_results(st.session_state.results, extract_keywords)
    
    with tab2:
        display_history()
    
    with tab3:
        display_about()

def process_file(uploaded_file, file_processor, summarizer, keyword_extractor, 
                model_choice, extract_keywords, num_keywords, max_length, min_length):
    """Process a single uploaded file"""
    try:
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_path = f"uploads/{file_id}_{uploaded_file.name}"
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text
        extracted_text = file_processor.extract_text(file_path, uploaded_file.type)
        
        if not extracted_text.strip():
            st.error(f"No text could be extracted from {uploaded_file.name}")
            return None
        
        # Generate summary
        summary = summarizer.summarize(
            extracted_text, 
            model_name=model_choice,
            max_length=max_length,
            min_length=min_length
        )
        
        # Extract keywords
        keywords = []
        if extract_keywords:
            keywords = keyword_extractor.extract_keywords(extracted_text, num_keywords)
        
        # Prepare result
        result = {
            "id": file_id,
            "filename": uploaded_file.name,
            "file_type": uploaded_file.type,
            "original_text": extracted_text,
            "summary": summary,
            "keywords": keywords,
            "model_used": model_choice,
            "timestamp": datetime.now().isoformat(),
            "text_length": len(extracted_text.split()),
            "summary_length": len(summary.split()) if summary else 0
        }
        
        # Save metadata
        save_metadata(result)
        
        return result
        
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return None

def process_text(raw_text, summarizer, keyword_extractor, model_choice, 
                extract_keywords, num_keywords, max_length, min_length):
    """Process raw text input"""
    try:
        # Generate summary
        summary = summarizer.summarize(
            raw_text,
            model_name=model_choice,
            max_length=max_length,
            min_length=min_length
        )
        
        # Extract keywords
        keywords = []
        if extract_keywords:
            keywords = keyword_extractor.extract_keywords(raw_text, num_keywords)
        
        # Prepare result
        result = {
            "id": str(uuid.uuid4()),
            "filename": "Direct Text Input",
            "file_type": "text/plain",
            "original_text": raw_text,
            "summary": summary,
            "keywords": keywords,
            "model_used": model_choice,
            "timestamp": datetime.now().isoformat(),
            "text_length": len(raw_text.split()),
            "summary_length": len(summary.split()) if summary else 0
        }
        
        # Save metadata
        save_metadata(result)
        
        return result
        
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return None

def display_results(results, extract_keywords):
    """Display processing results"""
    st.subheader("📋 Processing Results")
    
    for i, result in enumerate(results, 1):
        with st.expander(f"📄 {result['filename']} - Summary #{i}", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**📝 Summary:**")
                st.write(result['summary'])
                
                if extract_keywords and result['keywords']:
                    st.write("**🔑 Key Terms:**")
                    keywords_str = " • ".join([f"**{kw}**" for kw in result['keywords']])
                    st.markdown(keywords_str)
            
            with col2:
                st.write("**📊 Statistics:**")
                st.write(f"• Original: {result['text_length']} words")
                st.write(f"• Summary: {result['summary_length']} words")
                st.write(f"• Compression: {((result['text_length'] - result['summary_length']) / result['text_length'] * 100):.1f}%")
                st.write(f"• Model: {result['model_used'].split('/')[-1]}")
                
                # Export options
                st.write("**💾 Export Options:**")
                export_manager = ExportManager()
                
                # Generate exports
                pdf_path = export_manager.export_to_pdf(result)
                txt_path = export_manager.export_to_txt(result)
                
                # Download buttons
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📄 Download PDF",
                        data=pdf_file.read(),
                        file_name=f"summary_{result['filename']}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                with open(txt_path, "r", encoding="utf-8") as txt_file:
                    st.download_button(
                        label="📝 Download TXT",
                        data=txt_file.read(),
                        file_name=f"summary_{result['filename']}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

def display_history():
    """Display processing history"""
    st.subheader("📊 Processing History")
    
    metadata = load_metadata()
    if not metadata:
        st.info("No processing history available yet.")
        return
    
    # Display recent summaries
    for item in reversed(metadata[-10:]):  # Show last 10 items
        with st.expander(f"📄 {item['filename']} - {item['timestamp'][:19]}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Summary:**")
                st.write(item['summary'])
                
                if item.get('keywords'):
                    st.write("**Keywords:**")
                    st.write(" • ".join(item['keywords']))
            
            with col2:
                st.write("**Details:**")
                st.write(f"• Model: {item['model_used']}")
                st.write(f"• Original: {item['text_length']} words")
                st.write(f"• Summary: {item['summary_length']} words")

def display_about():
    """Display about information"""
    st.subheader("ℹ️ About AI-Powered Notes Summarizer")
    
    st.markdown("""
    ### 🎯 Features
    - **Multi-format Support**: Upload PDF, TXT, or DOCX files
    - **AI Summarization**: Choose from BART, T5, or Pegasus models
    - **Smart Keyword Extraction**: Automatic key term identification
    - **Adaptive Length**: Medium summary per page (automatic sizing)
    - **Export Options**: Download summaries as PDF or TXT
    - **Processing History**: Track your summarization tasks
    
    ### 🛠️ Technologies Used
    - **Frontend**: Streamlit
    - **NLP Models**: HuggingFace Transformers (BART, T5, Pegasus)
    - **Keyword Extraction**: KeyBERT + spaCy
    - **File Processing**: PyPDF2, python-docx
    - **Export**: FPDF, ReportLab
    
    ### 📝 Model Information
    - **BART-Large-CNN**: Optimized for news and article summarization
    - **T5-Small**: Versatile text-to-text transformer (faster inference)
    - **Pegasus-XSUM**: Specialized for abstractive summarization
    
    ### 💡 Tips for Best Results
    - Upload clear, well-formatted documents
    - For technical content, try T5-Small
    - For news/articles, use BART-Large-CNN
    - For creative summaries, use Pegasus-XSUM
    """)

if __name__ == "__main__":
    main()





