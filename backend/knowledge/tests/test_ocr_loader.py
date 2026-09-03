from knowledge.storage.ocr_loader import OCRLoader

loader = OCRLoader()

document = loader.load(
    "knowledge/documents/linux/Linux.pdf"
)

print("=" * 60)
print("OCR TEST")
print("=" * 60)

print("Document:", document.name)
print("Characters:", len(document.content))

print("\nFirst 1000 Characters:\n")

print(document.content[:1000])