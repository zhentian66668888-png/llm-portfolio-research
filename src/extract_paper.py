"""Extract text from a user-supplied research PDF; no PDF is bundled."""
import argparse
from pathlib import Path
from pypdf import PdfReader


def extract_text(pdf_path, output_path):
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    Path(output_path).write_text(text, encoding="utf-8")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=Path("paper_text.txt"))
    args = parser.parse_args()
    extract_text(args.pdf, args.output)
    print(f"Paper text extracted to {args.output}")


if __name__ == "__main__":
    main()
