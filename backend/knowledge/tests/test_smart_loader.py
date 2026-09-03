from knowledge.storage.smart_loader import SmartLoader

loader = SmartLoader()

document = loader.load(

    "knowledge/documents/linux/Linux.pdf"

)

print("=" * 60)

print("SMART LOADER TEST")

print("=" * 60)

print()

print("Document:", document.name)

print("Type:", document.document_type)

print("Characters:", len(document.content))

print()

print(document.content[:1000])