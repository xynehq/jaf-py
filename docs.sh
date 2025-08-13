#!/bin/bash

# JAF Python Documentation Helper Script
# This script provides easy commands for working with the documentation

MKDOCS_CMD="python3 -m mkdocs"

# Check if mkdocs is installed
if ! python3 -c "import mkdocs" &> /dev/null; then
    echo "❌ MkDocs is not installed!"
    echo "📦 Install dependencies first:"
    echo "   pip install -r requirements-docs.txt"
    exit 1
fi

case "$1" in
    "build")
        echo "🔨 Building documentation..."
        $MKDOCS_CMD build
        ;;
    "serve")
        echo "🚀 Starting development server..."
        echo "🌐 Visit: http://127.0.0.1:8000"
        echo "⚠️  Press Ctrl+C to stop"
        $MKDOCS_CMD serve
        ;;
    "deploy")
        echo "🚀 Deploying to GitHub Pages..."
        $MKDOCS_CMD gh-deploy
        echo "✅ Deployed to: https://xynehq.github.io/jaf-py/"
        ;;
    "clean")
        echo "🧹 Cleaning build directory..."
        rm -rf site/
        echo "✅ Cleaned"
        ;;
    *)
        echo "JAF Python Documentation Helper"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  build   - Build the documentation"
        echo "  serve   - Start local development server"
        echo "  deploy  - Deploy to GitHub Pages"
        echo "  clean   - Clean build directory"
        echo ""
        echo "Examples:"
        echo "  ./docs.sh build   # Build docs"
        echo "  ./docs.sh serve   # Start dev server"
        echo "  ./docs.sh deploy  # Deploy to GitHub"
        echo ""
        ;;
esac