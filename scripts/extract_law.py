from pathlib import Path


def extract_from_pdf(pdf_path: Path) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return "\n\n".join(pages)


def extract_from_docx(docx_path: Path) -> str:
    from docx import Document

    doc = Document(str(docx_path))
    return "\n".join([p.text for p in doc.paragraphs])


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    out = data / "law_raw.txt"

    pdf = data / "law.pdf"
    docx = data / "law.docx"

    if pdf.exists():
        text = extract_from_pdf(pdf)
    elif docx.exists():
        text = extract_from_docx(docx)
    else:
        raise FileNotFoundError(
            "Put law.pdf or law.docx inside /data (named exactly law.pdf or law.docx)"
        )

    out.write_text(text, encoding="utf-8")
    print("Saved:", out)


if __name__ == "__main__":
    main()
