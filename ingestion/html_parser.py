from ingestion.base import BaseParser
import logging

logger = logging.getLogger(__name__)


class HTMLParser(BaseParser):
    """HTML parser using Trafilatura with BeautifulSoup fallback."""

    @staticmethod
    def accepts_extension(ext: str) -> bool:
        return ext.lower() in [".html", ".htm"]

    def extract(self, file_path: str) -> str:
        try:
            import trafilatura

            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
            text = trafilatura.extract(html, include_images=True, output_format="text")
            return text or ""
        except ImportError:
            logger.warning("Trafilatura not available, using BeautifulSoup fallback")
            return self._fallback_extract(file_path)

    def _fallback_extract(self, file_path: str) -> str:
        from bs4 import BeautifulSoup

        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
