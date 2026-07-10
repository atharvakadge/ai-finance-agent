"""
PDF Processing Service - Extracts and chunks text from PDF documents.

This is Step 1 of our RAG pipeline:
    PDF → Extract text → Split into chunks → (next: embed and store)

Why a separate service?
- Single Responsibility: this file ONLY handles PDF processing
- The LLM service doesn't need to know about PDFs
- The vector DB service (coming next) doesn't need to know about PDFs
- Each service does ONE job well
"""

import fitz  # PyMuPDF - imported as 'fitz' for historical reasons
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFService:
    """Handles PDF text extraction and chunking."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the PDF service.

        Args:
            chunk_size: Max characters per chunk.
                - Too small (200): loses context, chunks are meaningless
                - Too big (5000): embeddings become vague, retrieval is imprecise
                - Sweet spot (1000): enough context, focused meaning

            chunk_overlap: Characters shared between consecutive chunks.
                - Prevents important info from being split across chunks
                - 200 chars ≈ 1-2 sentences of overlap
                - Too much overlap = duplicate data, wasted storage
                - Too little = risk losing context at boundaries
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # What does 'separators' mean?
            # The splitter tries to split at the FIRST separator that works.
            # It tries "\n\n" (paragraph breaks) first - best split point.
            # If a chunk is still too big, it tries "\n" (line breaks).
            # Then ". " (sentences), then " " (words), then "" (characters).
            # This is the "recursive" part - it cascades through separators.
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,  # measure chunk size by character count
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.

        Args:
            pdf_path: Path to the PDF file on disk

        Returns:
            All text from the PDF as a single string

        How PyMuPDF works:
        - Opens the PDF and reads it page by page
        - Extracts text while preserving reading order
        - Handles multi-column layouts better than most libraries
        - Returns plain text (no formatting, no images)
        """
        doc = fitz.open(pdf_path)
        text = ""

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()

        doc.close()
        return text

    def extract_text_from_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes (for file uploads via API).

        When a user uploads a PDF through the API, we receive
        raw bytes, not a file path. This method handles that.

        Args:
            pdf_bytes: Raw PDF file content as bytes

        Returns:
            All text from the PDF as a single string
        """
        # stream=pdf_bytes tells PyMuPDF to read from memory, not disk
        # filetype="pdf" tells it to expect PDF format
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()

        doc.close()
        return text

    def chunk_text(self, text: str) -> list[dict]:
        """
        Split text into overlapping chunks for RAG.

        Args:
            text: The full document text

        Returns:
            List of chunk dictionaries, each containing:
            - 'content': the chunk text
            - 'chunk_index': position in the document (0, 1, 2, ...)
            - 'char_start': starting character position in original text

        Why return dicts instead of just strings?
        - We need metadata for each chunk (position, source page, etc.)
        - When the user asks a question and we retrieve a chunk,
          we need to tell them WHERE in the document it came from
        - This is how citations work in RAG
        """
        # create_documents expects a list of texts
        # It returns LangChain Document objects
        documents = self.text_splitter.create_documents([text])

        chunks = []
        for i, doc in enumerate(documents):
            chunks.append({
                "content": doc.page_content,
                "chunk_index": i,
            })

        return chunks

    def process_pdf(self, pdf_path: str = None, pdf_bytes: bytes = None) -> list[dict]:
        """
        Complete pipeline: PDF → text → chunks.

        This is the main method other services will call.
        Give it a PDF (path or bytes), get back chunks ready for embedding.

        Args:
            pdf_path: Path to PDF file (for local files)
            pdf_bytes: Raw PDF bytes (for uploads)
            One of these must be provided.

        Returns:
            List of chunk dicts ready for the next step (embedding)
        """
        # Validate input - one of the two must be provided
        if pdf_path is None and pdf_bytes is None:
            raise ValueError("Either pdf_path or pdf_bytes must be provided")

        # Step 1: Extract text
        if pdf_path:
            raw_text = self.extract_text_from_pdf(pdf_path)
        else:
            raw_text = self.extract_text_from_bytes(pdf_bytes)

        # Step 2: Basic cleaning
        # Annual reports have lots of extra whitespace, repeated headers, etc.
        cleaned_text = self._clean_text(raw_text)

        # Step 3: Chunk the text
        chunks = self.chunk_text(cleaned_text)

        return chunks

    def _clean_text(self, text: str) -> str:
        """
        Basic text cleaning for financial documents.

        Why clean?
        - PDFs have weird spacing from column layouts
        - Headers/footers repeat on every page
        - Extra whitespace wastes tokens and hurts embeddings
        """
        # Replace multiple newlines with double newline (paragraph break)
        import re
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Replace multiple spaces with single space
        text = re.sub(r" {2,}", " ", text)

        # Remove very short lines (likely headers/footers/page numbers)
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Keep lines that have meaningful content (more than 5 chars)
            # Very short lines are usually page numbers, headers, etc.
            if len(stripped) > 5 or stripped == "":
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)