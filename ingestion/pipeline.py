from ingestion.base import BaseParser, ChunkRecord
from ingestion.pdf_parser import PDFParser
from ingestion.html_parser import HTMLParser
from ingestion.text_cleaner import TextCleaner
from ingestion.chunker import Chunker

import logging
import os
from typing import List

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "microsoft/harrier-oss-v1-0.6b",
        chunking_strategy: str = "fixed",
        pdf_parser: str = "docling",
        html_parser: str = "docling",
    ):
        self.pdf_parser = pdf_parser
        self.html_parser = html_parser
        self.chunking_strategy = chunking_strategy
        self.chunker = Chunker(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def process_file(self, file_path: str, doc_id: str = None) -> List[ChunkRecord]:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions, 
            RapidOcrOptions, 
            TableStructureOptions, 
            TableFormerMode
        )
        from pathlib import Path
        
        # 1. Primary Configuration: High quality with Table extraction (FAST mode for stability)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True # We want tables!
        pipeline_options.table_structure_options.mode = TableFormerMode.FAST
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False
        pipeline_options.ocr_options = RapidOcrOptions()

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        try:
            logger.info(f"Attempting structured conversion (Tables enabled) for: {file_path}")
            result = converter.convert(Path(file_path))
        except Exception as e:
            # 2. Fallback: If structured layout fails (e.g., tensor padding crash), retry in simple mode.
            if "layout failed" in str(e).lower() or "unable to create tensor" in str(e).lower():
                logger.warning(f"Structured layout failed for {file_path}. Retrying in simple text-only mode...")
                pipeline_options.do_table_structure = False
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
                result = converter.convert(Path(file_path))
            else:
                raise e

        doc = result.document

        if doc_id is None:
            doc_id = os.path.splitext(os.path.basename(file_path))[0]

        # Pass the DoclingDocument and filename to the HierarchicalChunker
        chunks = self.chunker.chunk(doc, filename=file_path, doc_id=doc_id)

        return chunks

    def process_directory(
        self, directory: str, extensions: List[str] = None
    ) -> List[ChunkRecord]:
        if extensions is None:
            extensions = [".pdf", ".html", ".htm"]

        all_chunks = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in extensions:
                    file_path = os.path.join(root, file)
                    try:
                        chunks = self.process_file(file_path)
                        all_chunks.extend(chunks)
                        logger.info(f"Processed {file_path}: {len(chunks)} chunks")
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")

        return all_chunks
