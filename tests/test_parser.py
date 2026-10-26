import pytest
from pathlib import Path
from src.parser import DocumentParser, DocumentSection
from config import SAMPLE_DOCS_DIR
from src.sample_data import generate_sample_documents

@pytest.fixture(scope="module")
def sample_files():
    generate_sample_documents(SAMPLE_DOCS_DIR)
    return {
        "pdf": SAMPLE_DOCS_DIR / "Academic_Regulations_Handbook_2024.pdf",
        "txt": SAMPLE_DOCS_DIR / "Examination_Notice_Fall2024.txt",
        "docx": SAMPLE_DOCS_DIR / "CS101_Data_Structures_Syllabus.docx",
        "xlsx": SAMPLE_DOCS_DIR / "Attendance_and_Grading_Matrix.xlsx",
        "pptx": SAMPLE_DOCS_DIR / "Campus_Facilities_and_Hostel_Guide.pptx",
    }

def test_parse_pdf(sample_files):
    pdf_path = sample_files["pdf"]
    sections = DocumentParser.parse_file(pdf_path, pdf_path.name)
    assert len(sections) >= 3, "PDF should have at least 3 pages/sections"
    assert sections[0].file_type == "pdf"
    assert "Grading System" in sections[0].text or "1.1 Grading Scale" in sections[0].text
    assert sections[1].section_index == 2
    assert "75%" in sections[1].text

def test_parse_txt(sample_files):
    txt_path = sample_files["txt"]
    sections = DocumentParser.parse_file(txt_path, txt_path.name)
    assert len(sections) >= 1
    assert "CONTROLLER OF EXAMINATIONS" in sections[0].text
    assert "November 5, 2024" in sections[0].text

def test_parse_excel(sample_files):
    xlsx_path = sample_files["xlsx"]
    sections = DocumentParser.parse_file(xlsx_path, xlsx_path.name)
    assert len(sections) == 2, "Excel should have 2 sheets"
    sheet_names = [s.metadata.get("sheet_name") for s in sections]
    assert "Grading_Scale" in sheet_names
    assert "Attendance_Rules" in sheet_names

def test_parse_pptx(sample_files):
    pptx_path = sample_files["pptx"]
    sections = DocumentParser.parse_file(pptx_path, pptx_path.name)
    assert len(sections) == 4, "PPTX should have 4 slides"
    assert sections[1].section_type == "slide"
    assert "08:00 AM to 10:00 PM" in sections[1].text

def test_parse_docx(sample_files):
    docx_path = sample_files["docx"]
    sections = DocumentParser.parse_file(docx_path, docx_path.name)
    assert len(sections) >= 1
    assert "CS101" in sections[0].text
    assert "Cormen" in sections[0].text
