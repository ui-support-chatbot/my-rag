import json
import os
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class FileState:
    path: str
    hash: str
    last_modified: float
    doc_id: str
    chunk_count: int = 0
    metadata: Dict = field(default_factory=dict)

class IngestionState:
    def __init__(self, state_path: str = "storage/ingestion_state.json"):
        self.state_path = Path(state_path)
        self.files: Dict[str, FileState] = {}
        self.load()

    def load(self):
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for path, info in data.items():
                        self.files[path] = FileState(**info)
                logger.info(f"Loaded ingestion state from {self.state_path} ({len(self.files)} files)")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

    def save(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {path: asdict(state) for path, state in self.files.items()}
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    @staticmethod
    def calculate_hash(file_path: str) -> str:
        """Calculate MD5 hash of a file."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_file_status(self, file_path: str) -> str:
        """
        Returns 'new', 'modified', or 'unchanged'.
        """
        abs_path = os.path.abspath(file_path)
        if abs_path not in self.files:
            return "new"
        
        saved_state = self.files[abs_path]
        current_hash = self.calculate_hash(file_path)
        
        if current_hash != saved_state.hash:
            return "modified"
        
        return "unchanged"

    def update_file(self, file_path: str, doc_id: str, chunk_count: int, metadata: Dict = None):
        abs_path = os.path.abspath(file_path)
        current_hash = self.calculate_hash(file_path)
        last_modified = os.path.getmtime(file_path)
        
        self.files[abs_path] = FileState(
            path=abs_path,
            hash=current_hash,
            last_modified=last_modified,
            doc_id=doc_id,
            chunk_count=chunk_count,
            metadata=metadata or {}
        )
        self.save()

    def get_all_ingested(self) -> List[FileState]:
        return list(self.files.values())
