#!/bin/bash
# Start AI-Powered Notes Summarizer
echo "🚀 Starting AI-Powered Notes Summarizer..."
echo "📝 Loading application at http://localhost:8501"
echo ""

# Run Streamlit
cd /app
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.runOnSave=true