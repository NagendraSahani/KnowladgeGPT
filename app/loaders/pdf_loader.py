from pathlib import Path
from pypdf import PdfReader


class PDFLoader:

    def load_pdf(self, pdf_path: str) -> str:

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"{pdf_path} not found")

        reader = PdfReader(pdf_path)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text