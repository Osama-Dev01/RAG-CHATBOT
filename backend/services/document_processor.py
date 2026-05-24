import logging
import re
from typing import List
from pathlib import Path

# LangChain imports
from langchain_core.documents import Document
from langchain_docling.loader import DoclingLoader, ExportType
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_docling.loader import DoclingLoader, ExportType
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend


# Your settings
from backend.config.settings import settings
from docling.datamodel.pipeline_options import PdfPipelineOptions





logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        # Initialize the splitter using settings from your config
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n## ", "\n\n### ", "\n\n", "\n", " ", ""]
        )

    def process(self, file_path: str) -> List[Document]:
        """
        Main pipeline: Extract -> Clean -> Chunk
        """

        logger.info(f"Starting processing for: {file_path}")
        
      
        docs = self._extract_text(file_path)
        
       
        cleaned_docs = self._clean_documents(docs)
        
       
        chunks = self._chunk_documents(cleaned_docs)
        
        return chunks

    def _extract_text(self, file_path: str) -> List[Document]:
        """Extract text using Docling and return as Markdown Documents."""
       
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False
        
      
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend  
                )
            }
        )

        loader = DoclingLoader(
            file_path=file_path, 
            export_type=ExportType.MARKDOWN,
            converter=converter
        )
        docs = loader.load()
        
        # --- QUALITY CHECK: Extraction ---
        print("\n" + "="*60)
        print(f" EXTRACTION QUALITY CHECK: {Path(file_path).name}")
        print("="*60)
        for i, doc in enumerate(docs):
            # Print the first 500 characters on a single line for easy reading
            preview = doc.page_content[:500].replace("\n", " \\n ")
            print(f"--- Doc {i+1} Preview (First 500 chars) ---")
            print(f"{preview}...\n")
            
        return docs

    def _clean_documents(self, docs: List[Document]) -> List[Document]:
        """Clean the extracted Markdown text to improve embedding quality."""
        cleaned_docs = []
        for doc in docs:
            text = doc.page_content
            
          
            text = text.replace("\u200b", "")
            
         
            text = re.sub(r'\n{3,}', '\n\n', text)
            
           
            text = re.sub(r'[ \t]+\n', '\n', text)
            
           
            cleaned_doc = Document(page_content=text, metadata=doc.metadata)
            cleaned_docs.append(cleaned_doc)
            
        return cleaned_docs

    def _chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Chunk the cleaned documents and print a quality report."""
        chunks = self.text_splitter.split_documents(docs)
        
        #  Chunking ---
        print("\n" + "="*60)
        print(f" CHUNKING QUALITY CHECK: Created {len(chunks)} chunks")
        print(f" Settings: Size={settings.chunk_size}, Overlap={settings.chunk_overlap}")
        print("="*60)
        
       
        for i, chunk in enumerate(chunks[:3]):
            print(f"--- Chunk {i+1} [Size: {len(chunk.page_content)} chars] ---")
         
            print(f"Metadata: {chunk.metadata}") 
            print(f"Content:\n{chunk.page_content}\n")
            print("-" * 60)
            
        return chunks


document_processor = DocumentProcessor()