#!/bin/bash
# Quick start script for the Startup Name Brainstorming Agent

echo "🚀 Starting Startup Name Brainstormer..."
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "Please create a .env file with your API keys:"
    echo "  OPENAI_API_KEY=your_key_here"
    echo "  or"
    echo "  ANTHROPIC_API_KEY=your_key_here"
    echo ""
fi

# Check if dependencies are installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

# Run Streamlit app
echo "🌐 Launching Streamlit UI..."
python3 -m streamlit run app.py

