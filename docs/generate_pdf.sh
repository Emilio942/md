#!/bin/bash
# PDF export script for ProteinMD User Manual

echo "Generating PDF version of ProteinMD User Manual..."

# Check if sphinx is available
if ! command -v sphinx-build &> /dev/null; then
    echo "Error: Sphinx not found. Install with: pip install sphinx"
    exit 1
fi

# Check if LaTeX is available
if ! command -v pdflatex &> /dev/null; then
    echo "Warning: pdflatex not found. Installing texlive-latex-base..."
    sudo apt-get update
    sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended
fi

# Navigate to docs directory
cd "$(dirname "$0")"
if [[ ! -f "conf.py" ]]; then
    echo "Error: Not in docs directory. Please run from docs/ folder."
    exit 1
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf _build/

# Build LaTeX version
echo "Building LaTeX documentation..."
sphinx-build -b latex . _build/latex

# Check if LaTeX build succeeded
if [[ ! -f "_build/latex/ProteinMD.tex" ]]; then
    echo "Error: LaTeX build failed"
    exit 1
fi

# Convert to PDF
echo "Converting to PDF..."
cd _build/latex
pdflatex ProteinMD.tex
pdflatex ProteinMD.tex  # Run twice for proper cross-references
makeindex ProteinMD.idx
pdflatex ProteinMD.tex  # Final run with index

# Check if PDF was created
if [[ -f "ProteinMD.pdf" ]]; then
    echo "✅ PDF successfully created: _build/latex/ProteinMD.pdf"
    
    # Copy to user_guide directory for easy access
    cp ProteinMD.pdf ../../../user_guide/ProteinMD_User_Manual.pdf
    echo "✅ PDF copied to: user_guide/ProteinMD_User_Manual.pdf"
    
    # Show file size
    size=$(du -h ProteinMD.pdf | cut -f1)
    echo "📄 PDF size: $size"
else
    echo "❌ Error: PDF generation failed"
    exit 1
fi

echo "📚 User Manual PDF export complete!"
