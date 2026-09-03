from knowledge.vector_db.chroma_database import ChromaDatabase

db = ChromaDatabase()

collection = db.create_collection("test")

print("Collection Created")

print(collection.name)