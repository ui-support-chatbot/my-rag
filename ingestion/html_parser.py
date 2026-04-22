from ingestion.base import BaseParser
import logging

logger = logging.getLogger(__name__)


class HTMLParser(BaseParser):
    """HTML parser using Trafilatura with BeautifulSoup fallback."""

    @staticmethod
    def accepts_extension(ext: str) -> bool:
        return ext.lower() in [".html", ".htm"]

    def extract(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()

        trafilatura_error = None
        try:
            import trafilatura
        except ImportError as e:
            trafilatura_error = e
            logger.warning("Trafilatura not available for %s, using BeautifulSoup fallback", file_path)
        else:
            try:
                text = trafilatura.extract(
                    html,
                    include_images=True,
                    output_format="txt",
                )
            except Exception as e:
                trafilatura_error = e
                logger.warning(
                    "Trafilatura extraction failed for %s, using BeautifulSoup fallback: %s",
                    file_path,
                    e,
                )
            else:
                if text:
                    return text

                logger.warning(
                    "Trafilatura produced no text for %s, using BeautifulSoup fallback",
                    file_path,
                )

        try:
            return self._fallback_extract(html)
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract HTML text from {file_path}; "
                f"trafilatura_error={trafilatura_error!r}; "
                f"beautifulsoup_error={e!r}"
            ) from e

    def _fallback_extract(self, html: str) -> str:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "template"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
