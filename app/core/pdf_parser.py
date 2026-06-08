import logging
from io import BytesIO
from typing import List, Tuple

from PyPDF2 import PdfReader

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _extract_with_pypdf2(pdf_file) -> List[Tuple[str, dict]]:
    """Extract text per page using PyPDF2. Returns (text, metadata) tuples."""
    pages = []
    try:
        reader = PdfReader(pdf_file)
        filename = getattr(pdf_file, "name", "unknown.pdf")
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append((text, {"page": i + 1, "filename": filename, "source": filename}))
    except Exception as e:
        logger.error("PyPDF2 extraction failed: %s", e)
    return pages


def _extract_with_pymupdf(pdf_bytes: bytes, filename: str) -> List[Tuple[str, dict]]:
    """Extract text per page using PyMuPDF with OCR fallback for image pages."""
    pages = []
    try:
        import fitz  # pymupdf

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            text = page.get_text() or ""
            if settings.enable_ocr and len(text.strip()) < 50:
                try:
                    import pytesseract
                    from PIL import Image

                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        text = ocr_text
                        logger.info("OCR used for page %d of %s", i + 1, filename)
                except ImportError:
                    logger.debug("OCR dependencies not available")
                except Exception as e:
                    logger.warning("OCR failed for page %d: %s", i + 1, e)

            pages.append((text, {"page": i + 1, "filename": filename, "source": filename}))
        doc.close()
    except ImportError:
        logger.debug("PyMuPDF not available, skipping")
    except Exception as e:
        logger.error("PyMuPDF extraction failed: %s", e)
    return pages


def extract_pdf_pages(pdf_file, filename: str = "document.pdf") -> List[Tuple[str, dict]]:
    """Extract text from a PDF file object, returning per-page text and metadata."""
    if hasattr(pdf_file, "read"):
        pdf_file.seek(0)
        content = pdf_file.read()
        pdf_file.seek(0)
    else:
        content = pdf_file
        pdf_file = BytesIO(content)

    pages = _extract_with_pymupdf(content, filename)
    if not any(p[0].strip() for p in pages):
        pdf_file.seek(0) if hasattr(pdf_file, "seek") else None
        file_obj = BytesIO(content) if not hasattr(pdf_file, "read") else pdf_file
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        if not hasattr(file_obj, "name"):
            file_obj.name = filename
        pages = _extract_with_pypdf2(file_obj)

    return pages
