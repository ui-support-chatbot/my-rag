import re
from typing import List


class TextCleaner:
    @staticmethod
    def normalize(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    @staticmethod
    def remove_special_chars(text: str) -> str:
        text = re.sub(r"[^\w\s.,;:!?()-]", "", text)
        return text
