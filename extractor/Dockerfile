# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY epub_to_pdf.py .

# Create directories for input and output
RUN mkdir -p /app/input /app/output

# Set environment variables for paths
ENV INPUT_DIR=/app/input
ENV OUTPUT_DIR=/app/output
ENV EPUB_FILE=""
ENV OUTPUT_FILE=""

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Check if EPUB_FILE is set\n\
if [ -z "$EPUB_FILE" ]; then\n\
    echo "Error: EPUB_FILE environment variable is required"\n\
    echo "Usage: docker run -e EPUB_FILE=my_book.epub -v /path/to/epub:/app/input -v /path/to/output:/app/output epub-to-pdf"\n\
    exit 1\n\
fi\n\
\n\
# Set input file path\n\
INPUT_PATH="$INPUT_DIR/$EPUB_FILE"\n\
\n\
# Check if input file exists\n\
if [ ! -f "$INPUT_PATH" ]; then\n\
    echo "Error: EPUB file not found: $INPUT_PATH"\n\
    echo "Available files in input directory:"\n\
    ls -la "$INPUT_DIR"\n\
    exit 1\n\
fi\n\
\n\
# Set output file path\n\
if [ -n "$OUTPUT_FILE" ]; then\n\
    OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_FILE"\n\
else\n\
    # Generate output filename based on input\n\
    BASENAME=$(basename "$EPUB_FILE" .epub)\n\
    OUTPUT_PATH="$OUTPUT_DIR/${BASENAME}_structure.pdf"\n\
fi\n\
\n\
echo "Processing EPUB: $INPUT_PATH"\n\
echo "Output PDF: $OUTPUT_PATH"\n\
\n\
# Run the Python script\n\
python epub_to_pdf.py "$INPUT_PATH" --output "$OUTPUT_PATH"\n\
\n\
echo "✅ PDF generated successfully: $OUTPUT_PATH"\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Labels for metadata
LABEL maintainer="EPUB Structure Extractor"
LABEL description="Convert EPUB table of contents to PDF structure document"
LABEL version="1.0"