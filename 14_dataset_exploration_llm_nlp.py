from pathlib import Path
from collections import Counter

# Optional readers
from pypdf import PdfReader
from pptx import Presentation


DATASET_DIR = Path(r"D:\dataset")


PDF_EXTENSIONS = {".pdf"}
PPT_EXTENSIONS = {".ppt", ".pptx"}
DOC_EXTENSIONS = {".doc", ".docx"}
TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif"
}


def human_readable_size(size_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]

    size = float(size_bytes)

    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def count_pdf_pages(file_path):
    try:
        reader = PdfReader(str(file_path))
        return len(reader.pages)

    except Exception as error:
        print(f"[PDF ERROR] {file_path.name}: {error}")
        return 0


def count_pptx_slides(file_path):
    # python-pptx supports .pptx, not old binary .ppt
    if file_path.suffix.lower() != ".pptx":
        return 0

    try:
        presentation = Presentation(str(file_path))
        return len(presentation.slides)

    except Exception as error:
        print(f"[PPTX ERROR] {file_path.name}: {error}")
        return 0


def explore_dataset(dataset_dir):
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}"
        )

    all_files = [
        file
        for file in dataset_dir.rglob("*")
        if file.is_file()
    ]

    total_size = sum(
        file.stat().st_size
        for file in all_files
    )

    extension_counter = Counter(
        file.suffix.lower()
        for file in all_files
    )

    pdf_files = [
        f for f in all_files
        if f.suffix.lower() in PDF_EXTENSIONS
    ]

    ppt_files = [
        f for f in all_files
        if f.suffix.lower() in PPT_EXTENSIONS
    ]

    doc_files = [
        f for f in all_files
        if f.suffix.lower() in DOC_EXTENSIONS
    ]

    text_files = [
        f for f in all_files
        if f.suffix.lower() in TEXT_EXTENSIONS
    ]

    image_files = [
        f for f in all_files
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    other_files = [
        f for f in all_files
        if (
            f.suffix.lower() not in PDF_EXTENSIONS
            and f.suffix.lower() not in PPT_EXTENSIONS
            and f.suffix.lower() not in DOC_EXTENSIONS
            and f.suffix.lower() not in TEXT_EXTENSIONS
            and f.suffix.lower() not in IMAGE_EXTENSIONS
        )
    ]

    print("\n===================================")
    print("DATASET EXPLORATION")
    print("===================================")

    print(f"Dataset path: {dataset_dir}")
    print(f"Total files: {len(all_files)}")
    print(f"Total size: {human_readable_size(total_size)}")

    print("\n===================================")
    print("FILE TYPE COUNTS")
    print("===================================")

    print(f"Number of PDFs: {len(pdf_files)}")
    print(f"Number of PPT/PPTX: {len(ppt_files)}")
    print(f"Number of DOC/DOCX: {len(doc_files)}")
    print(f"Number of TXT/MD: {len(text_files)}")
    print(f"Number of images: {len(image_files)}")
    print(f"Other files: {len(other_files)}")

    print("\n===================================")
    print("PDF PAGE COUNT")
    print("===================================")

    total_pdf_pages = 0

    for file in pdf_files:
        pages = count_pdf_pages(file)
        total_pdf_pages += pages

        print(
            f"{file.name:50s} "
            f"{pages:5d} pages"
        )

    print(
        f"\nApproximate total PDF pages: "
        f"{total_pdf_pages}"
    )

    print("\n===================================")
    print("PRESENTATION SLIDE COUNT")
    print("===================================")

    total_slides = 0
    old_ppt_count = 0

    for file in ppt_files:
        if file.suffix.lower() == ".ppt":
            old_ppt_count += 1

            print(
                f"{file.name:50s} "
                f"OLD .ppt - slide count not read"
            )

            continue

        slides = count_pptx_slides(file)
        total_slides += slides

        print(
            f"{file.name:50s} "
            f"{slides:5d} slides"
        )

    print(
        f"\nApproximate total PPTX slides: "
        f"{total_slides}"
    )

    if old_ppt_count:
        print(
            f"Old .ppt files not counted: "
            f"{old_ppt_count}"
        )

    print("\n===================================")
    print("EXTENSION BREAKDOWN")
    print("===================================")

    for extension, count in sorted(
        extension_counter.items(),
        key=lambda item: item[0]
    ):
        extension_name = extension or "[no extension]"

        print(
            f"{extension_name:12s}: {count}"
        )

    if other_files:
        print("\n===================================")
        print("OTHER FILES")
        print("===================================")

        for file in other_files:
            print(file)

    print("\n===================================")
    print("SUMMARY")
    print("===================================")

    print(f"Number of PDFs: {len(pdf_files)}")
    print(f"Number of PPT/PPTX: {len(ppt_files)}")
    print(f"Number of DOC/DOCX: {len(doc_files)}")
    print(f"Number of TXT/MD: {len(text_files)}")
    print(f"Number of images: {len(image_files)}")

    print(
        f"Total size: "
        f"{human_readable_size(total_size)}"
    )

    print(
        f"Approximate PDF pages: "
        f"{total_pdf_pages}"
    )

    print(
        f"Approximate PPTX slides: "
        f"{total_slides}"
    )


if __name__ == "__main__":
    explore_dataset(DATASET_DIR)