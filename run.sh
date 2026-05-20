#!/bin/bash

# Stream Lit AI Chat - Startup Script

echo "🚀 Neural Talk - Local AI Chat Interface"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "✅ Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q streamlit langchain-core langchain-ollama pypdf 2>/dev/null || pip install streamlit langchain-core langchain-ollama pypdf

echo ""
echo "⚠️  IMPORTANT: Make sure Ollama is running!"
echo ""
echo "   In another terminal, run:"
echo "   $ ollama serve"
echo ""
echo "   To pull a model:"
echo "   $ ollama pull deepseek-coder:1.3b"
echo ""
echo "========================================"
echo ""
echo "🎧 Starting Streamlit app..."
echo "   Open http://localhost:8501 in your browser"
echo ""

# Run the app
streamlit run app.py

