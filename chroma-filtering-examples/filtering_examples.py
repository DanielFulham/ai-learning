import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.Client()

collection = client.create_collection(
    name="filter_demo",
    metadata={"Description": "Used to demo filtering in ChromaDB"},
    configuration={
        "embedding_function": ef,
    },
)

print(f"Collection created: {collection.name}")

collection.add(
    documents=[
        "This is a document about LangChain",
        "This is a reading about LlamaIndex",
        "This is a book about Python",
        "This is a document about pandas",
        "This is another document about LangChain"
    ],
    metadatas=[
        {"source": "langchain.com", "version": 0.1},
        {"source": "llamaindex.ai", "version": 0.2},
        {"source": "python.org", "version": 0.3},
        {"source": "pandas.pydata.org", "version": 0.4},
        {"source": "langchain.com", "version": 0.5},
    ],
    ids=["id1", "id2", "id3", "id4", "id5"]
)

print("\n---\n")

# 🏷️ Filter using Metadata — exact match ($eq)

print(collection.get(
    where={"source": {"$eq": "langchain.com"}}
))

print("\n---\n")

# 🏷️ Filter using Metadata — $and with $eq and $lt

print(collection.get(
    where={
        "$and": [
            {"source": {"$eq": "langchain.com"}}, 
            {"version": {"$lt": 0.3}}
        ]
    }
))

print("\n---\n")

# 🏷️ Filter using Metadata — $and with $in and $lt

print(collection.get(
    where={
        "$and": [
            {"source": {"$in": ["langchain.com", "llamaindex.ai"]}}, 
            {"version": {"$lt": 0.3}}
        ]
    }
))

print("\n---\n")

# 📝 Filter using Document Content

print(collection.get(
    where_document={"$contains":"pandas"}
))

print("\n---\n")

# 🏷️ + 📝 Combine Metadata and Document Content Filters

print(collection.get(
    where={"version": {"$gt": 0.1}},
    where_document={
        "$or": [
            {"$contains": "LangChain"},
            {"$contains": "Python"}
        ]
    }
))
