"""
CodedTool implementation for writing text content into a standard PDF file.
"""

from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Dict
import re

from neuro_san.interfaces.coded_tool import CodedTool


class WritePdf(CodedTool):
    """
    CodedTool that takes text content and writes it into a standard, valid PDF document.
    """

    @staticmethod
    def _text_to_pdf_bytes(text: str) -> bytes:
        """
        Converts text into standard PDF-1.4 binary data with 45 lines per page,
        10pt Helvetica font, and clean margins.
        """
        lines = text.split("\n")
        lines_per_page = 45
        pages_lines = [lines[i:i + lines_per_page] for i in range(0, len(lines), lines_per_page)]
        if not pages_lines:
            pages_lines = [[]]

        num_pages = len(pages_lines)
        pdf = bytearray()
        pdf.extend(b"%PDF-1.4\n")

        font_f1_num = 3 + 2 * num_pages
        font_f2_num = 3 + 2 * num_pages + 1
        offsets = {}

        # 1: Catalog
        offsets[1] = len(pdf)
        pdf.extend(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

        # 2: Pages
        kids_str = " ".join([f"{3 + i} 0 R" for i in range(num_pages)])
        offsets[2] = len(pdf)
        pdf.extend(f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>\nendobj\n".encode("latin1"))

        # Page and Content objects
        content_streams = []
        for i, page_lines in enumerate(pages_lines):
            content_obj_num = 3 + num_pages + i
            offsets[3 + i] = len(pdf)
            pdf.extend(
                f"{3 + i} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_obj_num} 0 R /Resources << /Font << /F1 {font_f1_num} 0 R /F2 {font_f2_num} 0 R >> >> >>\nendobj\n"
                .encode("latin1")
            )

            stream = "BT 50 740 Td 13 TL "
            for line in page_lines:
                parts = re.split(r"([\u25a0\u2588]+)", line)
                line_cmd = []
                for part in parts:
                    if not part:
                        continue
                    if part[0] in ("\u25a0", "\u2588"):
                        # In ZapfDingbats, 'n' renders visually as solid black square and extracts as ■
                        squares_n = "n" * len(part)
                        line_cmd.append(f"/F2 10 Tf ({squares_n}) Tj")
                    else:
                        escaped = part.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                        escaped_latin = escaped.encode("latin1", errors="replace").decode("latin1")
                        line_cmd.append(f"/F1 10 Tf ({escaped_latin}) Tj")
                if line_cmd:
                    stream += " ".join(line_cmd) + " T* "
                else:
                    stream += "T* "
            stream += "ET"
            content_streams.append(stream.encode("latin1"))

        for i, stream_bytes in enumerate(content_streams):
            content_obj_num = 3 + num_pages + i
            offsets[content_obj_num] = len(pdf)
            pdf.extend(f"{content_obj_num} 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin1"))
            pdf.extend(stream_bytes)
            pdf.extend(b"\nendstream\nendobj\n")

        # Font F1: Helvetica
        offsets[font_f1_num] = len(pdf)
        pdf.extend(f"{font_f1_num} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode("latin1"))

        # Font F2: ZapfDingbats (renders 'n' as solid black square ■)
        offsets[font_f2_num] = len(pdf)
        pdf.extend(f"{font_f2_num} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /ZapfDingbats >>\nendobj\n".encode("latin1"))

        # xref table
        xref_offset = len(pdf)
        total_objects = font_f2_num + 1
        pdf.extend(f"xref\n0 {total_objects}\n0000000000 65535 f \n".encode("latin1"))
        for obj_id in range(1, total_objects):
            pdf.extend(f"{offsets[obj_id]:010d} 00000 n \n".encode("latin1"))

        pdf.extend(f"trailer\n<< /Size {total_objects} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin1"))
        return bytes(pdf)

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Writes text content to a destination PDF file.

        :param args: Dictionary containing:
                     - "file_path" (str, required): Path to output PDF file.
                     - "content" (str, required): Text content to write.
        :param sly_data: Agent hierarchy bulletin board (optional).
        :return: Dictionary containing output metadata.
        """
        logger: Logger = getLogger(self.__class__.__name__)
        file_path_str = args.get("file_path")
        content = args.get("content", "")

        if not file_path_str:
            raise ValueError("invalid_input: 'file_path' is required.")

        file_path = Path(file_path_str).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("WritePdf: generating PDF for %s (%d chars)", file_path, len(content))
        pdf_bytes = self._text_to_pdf_bytes(content)

        try:
            file_path.write_bytes(pdf_bytes)
        except Exception as exc:
            raise ValueError(f"write_error: Failed to write PDF to '{file_path}': {exc}") from exc

        logger.info("WritePdf: successfully wrote %d bytes to %s", len(pdf_bytes), file_path)

        return {
            "path": str(file_path),
            "format": "pdf",
            "bytes_written": len(pdf_bytes),
            "status": "success"
        }

