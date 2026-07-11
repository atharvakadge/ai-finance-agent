"""
PDF Processing Service - Extracts text with section detection.
"""

import re
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFService:

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        # Section markers found in Indian annual reports
        self.section_markers = {
            "consolidated": [
                "consolidated financial statements",
                "consolidated statement of profit and loss",
                "consolidated balance sheet",
                "consolidated cash flow",
            ],
            "standalone": [
                "standalone financial statements",
                "standalone statement of profit and loss",
                "standalone balance sheet",
            ],
            "management": [
                "management discussion and analysis",
                "management discussion & analysis",
                "directors' report",
                "director's report",
            ],
            "chairman": [
                "chairman's letter",
                "chairman and managing director",
                "letter to shareholders",
            ],
            "segment": [
                "segment reporting",
                "segment information",
                "business segment",
            ],
            "overview": [
                "financial highlights",
                "performance highlights",
                "at a glance",
                "key highlights",
            ],
        }

    def extract_text_from_bytes(self, pdf_bytes: bytes) -> list[dict]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text", sort=True)
            table_text = self._extract_tables_from_page(page)

            combined = text
            if table_text:
                combined += "\n\n[TABLE DATA]\n" + table_text

            if combined.strip():
                pages.append({
                    "text": combined,
                    "page": page_num + 1,
                })

        doc.close()
        return pages

    def _extract_tables_from_page(self, page) -> str:
        try:
            tables = page.find_tables()
            if not tables or len(tables.tables) == 0:
                return ""

            table_texts = []
            for table in tables:
                data = table.extract()
                if not data:
                    continue

                rows = []
                for row in data:
                    cleaned = [str(cell).strip() if cell else "" for cell in row]
                    if any(c for c in cleaned):
                        rows.append(" | ".join(cleaned))

                if rows:
                    table_texts.append("\n".join(rows))

            return "\n\n".join(table_texts)
        except Exception:
            return ""

    def _detect_section(self, text: str, current_section: str) -> str:
        """
        Detect which section of the annual report this text belongs to.

        Scans for section markers and returns the section name.
        If no marker found, inherits the current section (because
        pages within a section don't repeat the header).
        """
        text_lower = text.lower()

        for section_name, markers in self.section_markers.items():
            for marker in markers:
                if marker in text_lower:
                    return section_name

        return current_section

    def chunk_text(self, pages: list[dict]) -> list[dict]:
        chunks = []
        chunk_index = 0
        current_section = "unknown"

        for page_data in pages:
            page_num = page_data["page"]
            text = page_data["text"]

            # Detect section from page content
            current_section = self._detect_section(text, current_section)

            documents = self.text_splitter.create_documents([text])

            for doc in documents:
                chunks.append({
                    "content": doc.page_content,
                    "chunk_index": chunk_index,
                    "page": page_num,
                    "section": current_section,
                })
                chunk_index += 1

        return chunks

    def process_pdf(self, pdf_path: str = None, pdf_bytes: bytes = None) -> list[dict]:
        if pdf_path is None and pdf_bytes is None:
            raise ValueError("Either pdf_path or pdf_bytes must be provided")

        if pdf_bytes:
            pages = self.extract_text_from_bytes(pdf_bytes)
        else:
            doc = fitz.open(pdf_path)
            pages = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text", sort=True)
                table_text = self._extract_tables_from_page(page)
                combined = text
                if table_text:
                    combined += "\n\n[TABLE DATA]\n" + table_text
                if combined.strip():
                    pages.append({"text": combined, "page": page_num + 1})
            doc.close()

        cleaned_pages = []
        for page_data in pages:
            cleaned_pages.append({
                "text": self._clean_text(page_data["text"]),
                "page": page_data["page"],
            })

        chunks = self.chunk_text(cleaned_pages)
        return chunks

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 3 or stripped == "":
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)