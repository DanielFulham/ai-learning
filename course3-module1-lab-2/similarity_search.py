import chromadb
from chromadb.utils import embedding_functions

# Define the embedding function using SentenceTransformers
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create a new instance of ChromaClient to interact with the Chroma DB
client = chromadb.Client()

# Define the name for the collection to be created or retrieved
collection_name = "my_grocery_collection"

# Define the main function to interact with the Chroma DB
def main():
    try:
        # Create a collection with cosine distance metric
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "A collection for storing grocery data"},
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": ef
            }
        )
        print(f"Collection created: {collection.name}")

        # Array of grocery-related text items
        texts = [
            'fresh red apples',
            'organic bananas',
            'ripe mangoes',
            'whole wheat bread',
            'farm-fresh eggs',
            'natural yogurt',
            'frozen vegetables',
            'grass-fed beef',
            'free-range chicken',
            'fresh salmon fillet',
            'aromatic coffee beans',
            'pure honey',
            'golden apple',
            'red fruit'
        ]

        # Create unique IDs for each text item
        ids = [f"food_{index + 1}" for index, _ in enumerate(texts)]

        # Add documents to the collection
        collection.add(
            documents=texts,
            metadatas=[{"source": "grocery_store", "category": "food"} for _ in texts],
            ids=ids
        )

        # Retrieve and display all items in the collection
        all_items = collection.get()
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")

        # Function to perform a similarity search in the collection
        def perform_similarity_search(collection, all_items):
            try:
                # query_term = "apple"
                query_term=["red", "fresh"]

                if (isinstance(query_term, str)):
                    query_term = [query_term]

                results = collection.query(
                    query_texts=query_term,
                    n_results=3
                )

                if not results or not results['ids'] or len(results['ids'][0]) == 0:
                    print(f'No documents found similar to "{query_term}"')
                    return
                
                for q in range(len(query_term)):
                    print(f'Top 3 similar documents to "{query_term[q]}":')
                    for i in range(min(3, len(results['ids'][q]))):
                        doc_id = results['ids'][q][i]
                        score = results['distances'][q][i]
                        text = results['documents'][q][i]
                        if not text:
                            print(f' - ID: {doc_id}, Text: "Text not available", Score: {score:.4f}')
                        else:
                            print(f' - ID: {doc_id}, Text: "{text}", Score: {score:.4f}')
            except Exception as error:
                print(f"Error in similarity search: {error}")

        # Call the similarity search function
        perform_similarity_search(collection, all_items)

    except Exception as error:
        print(f"Error: {error}")

if __name__ == "__main__":
    main()