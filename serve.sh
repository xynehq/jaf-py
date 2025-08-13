#!/bin/bash

# JAF Python Documentation Development Server
# This script starts a local development server for the documentation

echo "🚀 Starting JAF Python Documentation Server..."
echo "📁 Serving from: $(pwd)"
echo "🌐 URL: http://127.0.0.1:8000"
echo ""
echo "💡 The server will automatically reload when you make changes to:"
echo "   - Documentation files (docs/*.md)"
echo "   - Configuration (mkdocs.yml)"
echo "   - Assets (docs/assets/*)"
echo ""
echo "⚠️  Press Ctrl+C to stop the server"
echo ""

# Check if mkdocs is installed
if ! command -v mkdocs &> /dev/null; then
    echo "❌ MkDocs is not installed!"
    echo "📦 Install it with: pip install -r requirements-docs.txt"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "mkdocs.yml" ]; then
    echo "❌ mkdocs.yml not found!"
    echo "📁 Please run this script from the project root directory"
    exit 1
fi

# Start the development server
mkdocs serve --dev-addr 127.0.0.1:8000