import json
import os
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FileFingerprint:
    path: str
    hash: str
    last_modified: float
    size: int
    legacy_md5: Optional[str] = None


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

                # Backward compatible with the previous path -> FileState JSON shape.
                file_data = data.get("files", data) if isinstance(data, dict) else {}
                for path, info in file_data.items():
                    if not isinstance(info, dict):
                        continue
                    self.files[path] = FileState(
                        path=info.get("path", path),
                        hash=info.get("hash", ""),
                        last_modified=info.get("last_modified", 0.0),
                        doc_id=info.get("doc_id", ""),
                        chunk_count=info.get("chunk_count", 0),
                        metadata=info.get("metadata") or {},
                    )
                logger.info(
                    f"Loaded ingestion state from {self.state_path} ({len(self.files)} files)"
                )
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
    def calculate_hash(file_path: str, algorithm: str = "sha256") -> str:
        """Calculate a content hash for exact byte-level duplicate detection."""
        hasher = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def scan_file(self, file_path: str) -> FileFingerprint:
        abs_path = os.path.abspath(file_path)
        state = self.files.get(abs_path)
        legacy_md5 = None

        # Old state used MD5. Compute it only when needed for compatibility.
        if (state and len(state.hash) == 32) or self._has_legacy_hashes():
            legacy_md5 = self.calculate_hash(file_path, "md5")

        return FileFingerprint(
            path=abs_path,
            hash=self.calculate_hash(file_path, "sha256"),
            last_modified=os.path.getmtime(file_path),
            size=os.path.getsize(file_path),
            legacy_md5=legacy_md5,
        )

    def _has_legacy_hashes(self) -> bool:
        return any(len(state.hash) == 32 for state in self.files.values())

    def _is_alias(self, state: FileState) -> bool:
        return state.metadata.get("status") == "duplicate" or bool(
            state.metadata.get("alias_of")
        )

    def find_canonical_by_hash(self, content_hash: str) -> Optional[FileState]:
        for state in self.files.values():
            if state.hash == content_hash and not self._is_alias(state):
                return state
        return None

    def classify_fingerprint(self, fingerprint: FileFingerprint) -> Dict:
        """
        Classify a scanned file as new, modified, unchanged, or duplicate.
        Returns a dict so pipeline snapshots can reuse the same metadata.
        """
        existing = self.files.get(fingerprint.path)
        hashes_match = existing and (
            existing.hash == fingerprint.hash
            or (
                fingerprint.legacy_md5 is not None
                and existing.hash == fingerprint.legacy_md5
            )
        )

        if hashes_match:
            if self._is_alias(existing):
                canonical = self.find_canonical_by_hash(fingerprint.hash)
                return {
                    "status": "duplicate",
                    "reason": "same content as canonical file",
                    "canonical": canonical,
                    "existing": existing,
                }
            return {
                "status": "unchanged",
                "reason": "same path and content hash",
                "canonical": existing,
                "existing": existing,
            }

        canonical = self.find_canonical_by_hash(fingerprint.hash)
        if not canonical and fingerprint.legacy_md5:
            canonical = self.find_canonical_by_hash(fingerprint.legacy_md5)
        if canonical and canonical.path != fingerprint.path:
            status = "duplicate"
            reason = "same content as canonical file"
        elif existing:
            status = "modified"
            reason = "same path with changed content"
        else:
            status = "new"
            reason = "new content hash"

        return {
            "status": status,
            "reason": reason,
            "canonical": canonical,
            "existing": existing,
        }

    def get_file_status(self, file_path: str) -> str:
        """Compatibility helper returning new, modified, unchanged, or duplicate."""
        fingerprint = self.scan_file(file_path)
        return self.classify_fingerprint(fingerprint)["status"]

    @staticmethod
    def doc_id_for_hash(content_hash: str) -> str:
        return f"doc_{content_hash[:12]}"

    def update_file(
        self,
        file_path: str,
        doc_id: str,
        chunk_count: int,
        metadata: Dict = None,
        fingerprint: Optional[FileFingerprint] = None,
        save: bool = True,
    ):
        fingerprint = fingerprint or self.scan_file(file_path)

        self.files[fingerprint.path] = FileState(
            path=fingerprint.path,
            hash=fingerprint.hash,
            last_modified=fingerprint.last_modified,
            doc_id=doc_id,
            chunk_count=chunk_count,
            metadata=metadata or {},
        )
        if save:
            self.save()

    def record_alias(
        self,
        file_path: str,
        canonical: FileState,
        fingerprint: Optional[FileFingerprint] = None,
        metadata: Dict = None,
        save: bool = True,
    ):
        fingerprint = fingerprint or self.scan_file(file_path)
        alias_metadata = {
            "status": "duplicate",
            "alias_of": canonical.path,
            "canonical_doc_id": canonical.doc_id,
            "canonical_path": canonical.path,
        }
        if metadata:
            alias_metadata.update(metadata)

        self.files[fingerprint.path] = FileState(
            path=fingerprint.path,
            hash=fingerprint.hash,
            last_modified=fingerprint.last_modified,
            doc_id=canonical.doc_id,
            chunk_count=0,
            metadata=alias_metadata,
        )
        if save:
            self.save()

    def get_all_ingested(self) -> List[FileState]:
        return list(self.files.values())
