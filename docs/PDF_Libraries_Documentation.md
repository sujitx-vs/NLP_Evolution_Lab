# PDF Reading Libraries Exploration

## Overview

This notebook explores multiple Python libraries for reading and
extracting content from PDF documents. The goal is to compare different
libraries based on their capabilities, ease of use, and suitability for
various PDF-processing tasks such as text extraction, table extraction,
OCR, and document conversion.

The notebook uses a sample PDF (`Sample01.pdf`) to demonstrate how each
library works.

## Libraries Covered

### 1. PyPDF

-   Read PDF files
-   Extract text
-   Access metadata
-   Get page count

### 2. PDFium (`pypdfium2`)

-   Render PDF pages
-   Convert pages into images

### 3. pdfplumber

-   Extract text
-   Extract tables

### 4. PyMuPDF4LLM

-   Convert PDFs into LLM-friendly text.

### 5. Unstructured

-   Parse complex PDFs.

### 6. Marker PDF

-   Convert PDFs into Markdown.

### 7. Textract

-   Extract text from PDFs and other document formats.

## Workflow

1.  Install libraries.
2.  Load PDF.
3.  Extract metadata and text.
4.  Compare outputs.
5.  Evaluate libraries.

## Conclusion

Choose the library based on your use case: PyPDF for simple extraction,
pdfplumber for tables, pypdfium2 for rendering, and
PyMuPDF4LLM/Unstructured/Marker PDF for AI workflows.
