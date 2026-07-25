from knowledge.storage.pdf_loader import PDFLoader

loader = PDFLoader()

document = loader.load(
    "knowledge/documents/linux/Linux.pdf"
)

print("=" * 50)
print("PDF LOADER TEST")
print("=" * 50)

print("Name :", document.name)
print("Type :", document.document_type)
print("Pages:", document.metadata["pages"])

print("\nFirst 300 Characters\n")

print(document.content[:300])