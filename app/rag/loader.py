"""
Document Loading Module

This module handles loading various document types (.pdf, .txt, .md, .docx)
into a unified format for processing.

Process:
1. Scan directory for supported document types
2. Extract text from each document type appropriately
3. Return standardized document objects with metadata
4. Handle errors gracefully for corrupted/unreadable files
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from PyPDF2 import PdfReader
import docx

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """
    Clean text by removing/replacing problematic characters.
    
    Handles:
    - Unicode surrogate characters
    - Control characters
    - Invalid UTF-8 sequences
    """
    if not text:
        return text
    
    # Remove surrogate characters
    try:
        text = text.encode('utf-8', 'replace').decode('utf-8', 'replace')
    except:
        pass
    
    # Replace common problematic characters
    replacements = {
        '\ud835': '',  # Mathematical Alphanumeric Symbols surrogate
        '\ufffd': '',  # Replacement character
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


class DocumentLoader:
    """
    Utility class for loading documents from various formats.
    Supports: PDF, TXT, Markdown, DOCX
    """
    
    SUPPORTED_FORMATS = {
        ".pdf": "load_pdf",
        ".txt": "load_text",
        ".md": "load_markdown",
        ".docx": "load_docx"
    }
    
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Extracted text content
        
        Raises:
            Exception: If PDF cannot be read
        """
        try:
            text_content = []
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            # Clean text to remove problematic characters
                            page_text = clean_text(page_text)
                            text_content.append(page_text)
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {file_path}: {e}")
                        continue
            
            result = "\n\n".join(text_content)
            logger.debug(f"Extracted {num_pages} pages from {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
    
    @staticmethod
    def load_text(file_path: str) -> str:
        """
        Load plain text file.
        
        Args:
            file_path: Path to text file
        
        Returns:
            File content as string
        
        Raises:
            Exception: If file cannot be read
        """
        try:
            content = Path(file_path).read_text(encoding='utf-8')
            logger.debug(f"Loaded text file: {file_path}")
            return content
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                content = Path(file_path).read_text(encoding='latin-1')
                logger.debug(f"Loaded text file with latin-1 encoding: {file_path}")
                return content
            except Exception as e:
                logger.error(f"Error loading text file {file_path}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            raise
    
    @staticmethod
    def load_markdown(file_path: str) -> str:
        """
        Load markdown file (treated as plain text).
        
        Args:
            file_path: Path to markdown file
        
        Returns:
            File content as string
        """
        try:
            content = Path(file_path).read_text(encoding='utf-8')
            logger.debug(f"Loaded markdown file: {file_path}")
            return content
        except Exception as e:
            logger.error(f"Error loading markdown file {file_path}: {e}")
            raise
    
    @staticmethod
    def load_docx(file_path: str) -> str:
        """
        Extract text from DOCX file.
        
        Args:
            file_path: Path to DOCX file
        
        Returns:
            Extracted text content
        
        Raises:
            Exception: If DOCX cannot be read
        """
        try:
            doc = docx.Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            content = "\n".join(paragraphs)
            logger.debug(f"Extracted {len(paragraphs)} paragraphs from {file_path}")
            return content
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
            raise
    
    @classmethod
    def load_document(cls, file_path: str) -> Dict[str, Any]:
        """
        Load a single document with appropriate loader.
        
        Args:
            file_path: Path to document file
        
        Returns:
            Dict with 'filename', 'text', 'source', and 'file_type'
        
        Raises:
            ValueError: If file type is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in cls.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file type: {file_extension}. "
                f"Supported types: {list(cls.SUPPORTED_FORMATS.keys())}"
            )
        
        # Get appropriate loader method
        loader_method_name = cls.SUPPORTED_FORMATS[file_extension]
        loader_method = getattr(cls, loader_method_name)
        
        # Load document
        text_content = loader_method(str(file_path))
        
        if not text_content or not text_content.strip():
            logger.warning(f"Loaded document is empty: {file_path}")
        
        return {
            "filename": file_path.name,
            "text": text_content,
            "source": str(file_path),
            "file_type": file_extension[1:],  # Remove leading dot
            "file_size_bytes": file_path.stat().st_size
        }
    
    @classmethod
    def load_documents_from_directory(
        cls,
        directory_path: str,
        file_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            file_types: Specific file types to load (e.g., ['.pdf', '.txt'])
        
        Returns:
            List of loaded documents with metadata
        """
        dir_path = Path(directory_path)
        
        if not dir_path.exists() or not dir_path.is_dir():
            logger.warning(f"Directory not found or not a directory: {directory_path}")
            return []
        
        if file_types is None:
            file_types = list(cls.SUPPORTED_FORMATS.keys())
        
        documents = []
        
        for file_type in file_types:
            for file_path in dir_path.glob(f"*{file_type}"):
                try:
                    doc = cls.load_document(str(file_path))
                    documents.append(doc)
                    logger.info(f"Loaded: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path.name}: {e}")
                    continue
        
        logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents


# Convenience functions
def load_document(file_path: str) -> Dict[str, Any]:
    """Load a single document"""
    return DocumentLoader.load_document(file_path)


def load_documents_from_directory(directory_path: str) -> List[Dict[str, Any]]:
    """Load all documents from directory"""
    return DocumentLoader.load_documents_from_directory(directory_path)
