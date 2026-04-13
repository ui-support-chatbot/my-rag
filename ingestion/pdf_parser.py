from ingestion.base import BaseParser
import logging

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """PDF parser using Docling with pymupdf fallback."""

    @staticmethod
    def accepts_extension(ext: str) -> bool:
        return ext.lower() == ".pdf"

    def extract(self, file_path: str) -> str:
        try:
            from docling.document_converter import DocumentConverter
            from pathlib import Path

            converter = DocumentConverter()
            result = converter.convert(source=Path(file_path))
            return result.document.export_to_markdown()

        except ImportError:
            logger.warning("Docling not available, using pymupdf fallback")
            return self._fallback_extract(file_path)

    def _fallback_extract(self, file_path: str) -> str:
        import pymupdf

        doc = pymupdf.Document(file_path)
        text_parts = [page.get_text() for page in doc]
        return "\n\n".join(text_parts)
