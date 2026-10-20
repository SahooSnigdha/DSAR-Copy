"""
CodedTool implementation for reading PDF documents and extracting text and metadata.
"""

from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Dict
import pypdf

from neuro_san.interfaces.coded_tool import CodedTool


class ReadPdf(CodedTool):
    """
    CodedTool that reads a PDF file and extracts text content and page structure.
    """

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reads a local PDF file and returns its extracted text and page information.

        :param args: Dictionary containing:
                     - "file_path" (str, required): Absolute or relative path to the PDF file.
                     - "allowed_paths" (list[str], optional): Whitelist of directories allowed to be read.
        :param sly_data: Agent hierarchy bulletin board (optional).
        :return: Dictionary containing:
                 - "path": The resolved absolute path.
                 - "format": "pdf"
                 - "total_pages": Total number of pages in the PDF.
                 - "content": Full text content across all pages.
                 - "pages": List of per-page objects with page_number and text.
                 - "char_count": Total character count.
        """
        logger: Logger = getLogger(self.__class__.__name__)
        file_path_str = args.get("file_path")
        if not file_path_str:
            raise ValueError("invalid_input: 'file_path' is required.")

        file_path = Path(file_path_str).resolve()
        logger.info("ReadPdf: reading %s", file_path)

        if not file_path.exists():
            raise ValueError(f"path_not_found: PDF file '{file_path}' does not exist.")
        if file_path.is_dir():
            raise ValueError(f"is_a_directory: '{file_path}' is a directory, not a file.")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"invalid_input: '{file_path}' does not have a .pdf extension.")

        # Path access checks if allowed_paths are supplied
        allowed_paths = args.get("allowed_paths")
        if allowed_paths:
            allowed = False
            for allowed_p in allowed_paths:
                resolved_allowed = Path(allowed_p).resolve()
                if file_path == resolved_allowed or resolved_allowed in file_path.parents:
                    allowed = True
                    break
            if not allowed:
                raise ValueError(f"path_not_allowed: '{file_path}' is outside allowed paths {allowed_paths}.")

        try:
            reader = pypdf.PdfReader(str(file_path))
        except Exception as exc:
            raise ValueError(f"read_error: Failed to parse PDF '{file_path}': {exc}") from exc

        total_pages = len(reader.pages)
        pages_list = []
        text_chunks = []

        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            pages_list.append({
                "page_number": idx + 1,
                "text": page_text,
                "char_count": len(page_text)
            })
            text_chunks.append(page_text)

        full_content = "\n\n".join(text_chunks)

        logger.info("ReadPdf: successfully read %d pages (%d chars) from %s", total_pages, len(full_content), file_path)

        return {
            "path": str(file_path),
            "file_name": file_path.name,
            "file_stem": file_path.stem,
            "format": "pdf",
            "total_pages": total_pages,
            "content": full_content,
            "pages": pages_list,
            "char_count": len(full_content)
        }

