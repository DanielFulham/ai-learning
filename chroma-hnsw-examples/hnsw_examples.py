# Setup
import chromadb
from chromadb.utils import embedding_functions
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Collection creation

client = chromadb.Client()

space = "cosine" # Distance metric for the HNSW index. "cosine" for text/RAG (direction only), "l2" for spatial data, "ip" for inner product. Cannot be changed after collection creation.

ef_search = 100 # Candidate list size during search. Higher = better recall, slower queries. Online cost — paid on every query. Default: 100.

ef_construction = 200 # Candidate list size during index build. Higher = better index quality, slower build time. Offline cost — paid once. Default: 100.

max_neighbors = 16 # Max connections per node in the graph. Higher = denser graph, better search quality, more memory. Cannot be changed after collection creation. Default: 16.


collection = client.create_collection(
    name="my_collection_name",
    metadata={"topic": "query testing"},
    configuration={
        "hnsw": {
            "space": space,
            "ef_search": ef_search,
            "ef_construction": ef_construction, 
            "max_neighbors": max_neighbors, 
        },
        "embedding_function": ef,
        }
    )

# Add Collection

collection.add(
    documents=[
        "Giant pandas are a bear species that lives in mountainous areas.",
        "A pandas DataFrame stores two-dimensional, tabular data",
        "I think everyone agrees that pandas are some of the cutest animals on the planet",
        "A direct comparison between pandas and polars indicates that polars is a more efficient library than pandas.",
    ],
    metadatas=[
        {"topic": "animals"},
        {"topic": "data analysis"},
        {"topic": "animals"},
        {"topic": "data analysis"},
    ],
    ids=["id1", "id2", "id3", "id4"]
)

print("\n---\n")

# Querying in Chroma DB

# Query match

print(collection.query(
    query_texts=["cats"],
    n_results=10,
))

print("\n---\n")

# Query closer match

print(collection.query(
    query_texts=["polar bear"],
    n_results=1,
))

print("\n---\n")

# Query with Metadata filter

print(collection.query(
    query_texts=["polar bear"],
    n_results=1,
    where={'topic': 'animals'}
))

print("\n---\n")

# Query with Metadata filter that excludes a term

print(collection.query(
    query_texts=["polar bear"],
    n_results=1,
    where_document={'$not_contains': 'library'}
))

print("\n---\n")

# Query with both Metadata filter and Document filter

print(collection.query(
    query_texts=["polar bear"],
    n_results=1,
    where={'topic': 'animals'},
    where_document={'$not_contains': 'library'}
))