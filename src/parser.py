import os
import io
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

@dataclass
class DocumentSection:
    """Represents a parsed section or page of a document before chunking."""
    text: str
    filename: str
    file_type: str
    section_type: str  # 'page', 'slide', 'sheet', 'paragraph'
    section_index: int  # 1-indexed (e.g. Page 1, Slide 1, Sheet 1)
    section_label: str  # e.g. "Page 1", "Slide 3: Campus Hostels", "Sheet: Grading Matrix"
    metadata: Dict[str, Any] = field(default_factory=dict)

class DocumentParser:
    """Multi-format institutional document parser supporting PDF, TXT, Excel, PPTX, and DOCX."""

    @classmethod
    def parse_file(cls, file_source: Union[str, Path, io.BytesIO], filename: str) -> List[DocumentSection]:
        """
        Parses a file from a filesystem path or in-memory BytesIO buffer.
        Returns a list of DocumentSection objects with rich metadata.
        """
        ext = Path(filename).suffix.lower()
        
        # Read bytes if string/Path
        if isinstance(file_source, (str, Path)):
            with open(file_source, "rb") as f:
                content_bytes = f.read()
            stream = io.BytesIO(content_bytes)
        else:
            # Already BytesIO or file-like
            file_source.seek(0)
            content_bytes = file_source.read()
            stream = io.BytesIO(content_bytes)

        if ext == ".pdf":
            return cls._parse_pdf(stream, filename)
        elif ext in [".txt", ".md"]:
            return cls._parse_txt(content_bytes, filename)
        elif ext in [".xlsx", ".xls"]:
            return cls._parse_excel(stream, filename)
        elif ext == ".csv":
            return cls._parse_csv(content_bytes, filename)
        elif ext == ".pptx":
            return cls._parse_pptx(stream, filename)
        elif ext == ".docx":
            return cls._parse_docx(stream, filename)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Allowed: PDF, TXT, XLSX, XLS, CSV, PPTX, DOCX")

    @classmethod
    def _parse_pdf(cls, stream: io.BytesIO, filename: str) -> List[DocumentSection]:
        sections: List[DocumentSection] = []
        # Try PyMuPDF first for high fidelity extraction
        try:
            import pymupdf
            stream.seek(0)
            doc = pymupdf.open(stream=stream.read(), filetype="pdf")
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text("text").strip()
                if text:
                    sections.append(DocumentSection(
                        text=text,
                        filename=filename,
                        file_type="pdf",
                        section_type="page",
                        section_index=page_idx + 1,
                        section_label=f"Page {page_idx + 1}",
                        metadata={"total_pages": len(doc), "page_number": page_idx + 1}
                    ))
            if sections:
                return sections
        except Exception:
            pass

        # Fallback to pypdf
        import pypdf
        stream.seek(0)
        reader = pypdf.PdfReader(stream)
        total_pages = len(reader.pages)
        for page_idx, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                sections.append(DocumentSection(
                    text=text,
                    filename=filename,
                    file_type="pdf",
                    section_type="page",
                    section_index=page_idx + 1,
                    section_label=f"Page {page_idx + 1}",
                    metadata={"total_pages": total_pages, "page_number": page_idx + 1}
                ))
        return sections

    @classmethod
    def _parse_txt(cls, content_bytes: bytes, filename: str) -> List[DocumentSection]:
        # Try utf-8 first, fallback to utf-8-sig or latin-1
        text = ""
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                text = content_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        text = text.strip()
        if not text:
            return []

        # Split into logical sections if divided by clear horizontal lines or large breaks
        raw_sections = [s.strip() for s in text.split("\n\n---\n\n") if s.strip()]
        if len(raw_sections) <= 1:
            raw_sections = [s.strip() for s in text.split("\n\n\n") if s.strip()]

        if len(raw_sections) <= 1:
            return [DocumentSection(
                text=text,
                filename=filename,
                file_type="txt",
                section_type="document",
                section_index=1,
                section_label="Full Document",
                metadata={"page_number": 1}
            )]

        results = []
        for idx, sec_text in enumerate(raw_sections, start=1):
            results.append(DocumentSection(
                text=sec_text,
                filename=filename,
                file_type="txt",
                section_type="section",
                section_index=idx,
                section_label=f"Section {idx}",
                metadata={"section_number": idx, "page_number": idx}
            ))
        return results

    @classmethod
    def _parse_excel(cls, stream: io.BytesIO, filename: str) -> List[DocumentSection]:
        import openpyxl
        stream.seek(0)
        wb = openpyxl.load_workbook(stream, data_only=True)
        sections: List[DocumentSection] = []

        for sheet_idx, sheet_name in enumerate(wb.sheetnames, start=1):
            sheet = wb[sheet_name]
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                continue

            # Identify headers
            headers = [str(cell) if cell is not None else f"Col_{i+1}" for i, cell in enumerate(rows[0])]
            
            data_lines = [f"### Sheet: {sheet_name}"]
            data_lines.append(f"Headers: {', '.join(headers)}")

            for row_num, row in enumerate(rows[1:], start=2):
                row_values = [str(val).strip() for val in row if val is not None and str(val).strip()]
                if not row_values:
                    continue
                # Format each non-empty row clearly
                row_str_items = []
                for header, val in zip(headers, row):
                    if val is not None and str(val).strip():
                        row_str_items.append(f"{header}: {str(val).strip()}")
                if row_str_items:
                    data_lines.append(f"[Row {row_num}] " + "; ".join(row_str_items))

            sheet_text = "\n".join(data_lines)
            if sheet_text.strip():
                sections.append(DocumentSection(
                    text=sheet_text,
                    filename=filename,
                    file_type="excel",
                    section_type="sheet",
                    section_index=sheet_idx,
                    section_label=f"Sheet: {sheet_name}",
                    metadata={
                        "sheet_name": sheet_name,
                        "sheet_index": sheet_idx,
                        "row_count": len(rows),
                        "page_number": sheet_idx
                    }
                ))
        return sections

    @classmethod
    def _parse_csv(cls, content_bytes: bytes, filename: str) -> List[DocumentSection]:
        text = ""
        for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                text = content_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        lines = [line for line in csv.reader(io.StringIO(text)) if line]
        if not lines:
            return []

        headers = lines[0]
        data_lines = [f"### Table File: {filename}", f"Headers: {', '.join(headers)}"]

        for row_num, row in enumerate(lines[1:], start=2):
            items = [f"{h}: {val.strip()}" for h, val in zip(headers, row) if val.strip()]
            if items:
                data_lines.append(f"[Row {row_num}] " + "; ".join(items))

        return [DocumentSection(
            text="\n".join(data_lines),
            filename=filename,
            file_type="csv",
            section_type="sheet",
            section_index=1,
            section_label="Table Data",
            metadata={"row_count": len(lines), "page_number": 1}
        )]

    @classmethod
    def _parse_pptx(cls, stream: io.BytesIO, filename: str) -> List[DocumentSection]:
        from pptx import Presentation
        stream.seek(0)
        prs = Presentation(stream)
        sections: List[DocumentSection] = []

        for slide_idx, slide in enumerate(prs.slides, start=1):
            slide_texts = []
            slide_title = f"Slide {slide_idx}"

            # Extract title if present
            if slide.shapes.title and slide.shapes.title.text.strip():
                slide_title = slide.shapes.title.text.strip()
                slide_texts.append(f"## Slide Title: {slide_title}")

            # Extract body text from shapes
            for shape in slide.shapes:
                if shape.has_text_frame and shape != slide.shapes.title:
                    shape_text = shape.text_frame.text.strip()
                    if shape_text:
                        slide_texts.append(shape_text)
                elif shape.has_table:
                    table = shape.table
                    table_rows = []
                    for row in table.rows:
                        row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_cells:
                            table_rows.append(" | ".join(row_cells))
                    if table_rows:
                        slide_texts.append("Table:\n" + "\n".join(table_rows))

            # Extract speaker notes if available
            try:
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
                    if notes:
                        slide_texts.append(f"Speaker Notes: {notes}")
            except Exception:
                pass

            full_slide_text = "\n\n".join(slide_texts).strip()
            if full_slide_text:
                sections.append(DocumentSection(
                    text=full_slide_text,
                    filename=filename,
                    file_type="pptx",
                    section_type="slide",
                    section_index=slide_idx,
                    section_label=f"Slide {slide_idx}: {slide_title}",
                    metadata={
                        "slide_number": slide_idx,
                        "slide_title": slide_title,
                        "page_number": slide_idx
                    }
                ))
        return sections

    @classmethod
    def _parse_docx(cls, stream: io.BytesIO, filename: str) -> List[DocumentSection]:
        import docx
        stream.seek(0)
        doc = docx.Document(stream)
        paragraphs_text = []

        for para in doc.paragraphs:
            p_text = para.text.strip()
            if p_text:
                paragraphs_text.append(p_text)

        for table_idx, table in enumerate(doc.tables, start=1):
            table_lines = [f"Table {table_idx}:"]
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    table_lines.append(" | ".join(cells))
            if len(table_lines) > 1:
                paragraphs_text.append("\n".join(table_lines))

        full_text = "\n\n".join(paragraphs_text).strip()
        if not full_text:
            return []

        # Split into approximate sections/pages if long
        return [DocumentSection(
            text=full_text,
            filename=filename,
            file_type="docx",
            section_type="document",
            section_index=1,
            section_label="Document Body",
            metadata={"page_number": 1}
        )]
