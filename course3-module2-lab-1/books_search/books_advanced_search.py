# Importing necessary modules from the chromadb package:
# chromadb is used to interact with the Chroma DB database,
# embedding_functions is used to define the embedding model
from itertools import count

import chromadb
from chromadb.utils import embedding_functions
from data.books import books
from infra.embedding import Embedding

ef = Embedding().get_embedding_function()

client = chromadb.Client()

collection_name = "book_collection"

def main():
    try:

        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "A collection for storing book data"},
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": ef,
            }
        )

        print(f"Collection created: {collection.name}")

        # Exercise 1: Implement Similarity Search for Book Recommendations
        # Create meaningful text documents for each book that combine title, description, themes, and setting information for semantic search.
        # Hint: Combine multiple book attributes into descriptive text documents.

        book_documents = []
        for book in books:
            document = f"{book['title']} with the description {book['description']} explores themes of {book['themes']} and is set in {book['setting']}."
            document += f" It is a {book['genre']} book published in {book['year']}."
            document += f" Written by {book['author']}."
            book_documents.append(document)
        
        collection.add(
            ids=[book["id"] for book in books],
            documents=book_documents,
            metadatas=[{
                "title": book["title"],
                "author": book["author"],
                "genre": book["genre"],
                "year": book["year"],
                "rating": book["rating"],
                "pages": book["pages"],
                "description": book["description"],
                "themes": book["themes"],
                "setting": book["setting"]
            } for book in books]
        )

        all_items = collection.get()

        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")

        perform_advanced_search(collection, all_items)
        
        pass

    except Exception as error:
        print(f"Error: {error}")

def perform_advanced_search(collection, all_items):
    try:

        print("=== Similarity Search Examples ===")

        # Similarity search for "magical fantasy adventure"

        print("\nSearch for 'magical fantasy adventure':")
        query_text = "magical fantasy adventure"
        results = collection.query(
            query_texts=[query_text],
            n_results=3
        )

        print(f"Query: {query_text}")

        for i, (doc_id, document, distance) in enumerate(zip(
            results['ids'][0], results["documents"][0], results["distances"][0]
        )):
            metadata = results['metadatas'][0][i]
            print(f"  {i+1}. {metadata['title']} ({doc_id}) - Distance: {distance:.4f}")
            print(f"     Document: {document[:100]}...")

        # Filter books by genre (Fantasy or Science Fiction)
        print("\nFilter for Fantasy or Science Fiction genre:")
        
        results = collection.get(
            where={"genre": {"$in": ["Fantasy", "Science Fiction"]}},
        )

        print(f"Found {len(results['ids'])} books in Fantasy or Science Fiction genre:")
        for i, doc_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            print(f"  {i+1}. {metadata['title']} ({metadata['genre']}) - ID: {doc_id}")

        # Filter books by rating (4.5 or higher)
        print("\nFilter for books with rating 4.5 or higher:")
        results = collection.get(
            where={"rating": {"$gte": 4.5}},
        )

        print(f"Found {len(results['ids'])} books with rating 4.5 or higher:")
        for i, doc_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            print(f"  {i+1}. {metadata['title']} (Rating: {metadata['rating']}) - ID: {doc_id}")

        # Combined search: Find highly-rated dystopian books with similarity search
        print("\nCombined search for highly-rated dystopian books:")
        query_text = "dystopian society with themes of surveillance and freedom"
        results = collection.query(
            query_texts=[query_text],
            n_results=5,
            where={
                "$and": [
                    {"genre": "Dystopian"},
                    {"rating": {"$gte": 4.2}}
                ]
            }
        )

        print(f"Query: {query_text} with genre filter 'Dystopian' and rating filter 4.2 or higher")
        
        for i, (doc_id, document, distance) in enumerate(zip(
            results['ids'][0], results["documents"][0], results["distances"][0]
        )):
            metadata = results['metadatas'][0][i]
            print(f"  {i+1}. {metadata['title']} ({doc_id}) - Distance: {distance:.4f} - Rating: {metadata['rating']}★")
            print(f"     Document: {document[:100]}...")

        # Books published in a specific decade
        print("\nFilter for books published in the 1950s:")
        results = collection.get(
            where={
                "$and": [
                    {"year": {"$gte": 1950}},
                    {"year": {"$lt": 1960}}
                ]
            }
        )

        count = len(results['ids'])
        print(f"Found {count} book{'s' if count != 1 else ''} published in the 1950s:")
        for i, doc_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            print(f"  {i+1}. {metadata['title']} (Published: {metadata['year']}) - ID: {doc_id}")

        # Books with similar page counts

        print("\nSearch for books with similar page counts:")
        target = 300
        tolerance = 100
        results = collection.get(
            where={
                "$and": [
                    {"pages": {"$gte": target - tolerance}},
                    {"pages": {"$lte": target + tolerance}}
                ]
            }
        )
        print(f"Found {len(results['ids'])} books with page counts between {target - tolerance} and {target + tolerance}:")
        
        for i, doc_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            print(f"  {i+1}. {metadata['title']} (Pages: {metadata['pages']}) - ID: {doc_id}")

        # Books that match multiple themes
        print("\nSearch for books that match multiple themes:")
        query_text = "books that explore themes of friendship, courage, and good vs evil"
        results = collection.query(
            query_texts=[query_text],
            n_results=5,
            where={
                "$or": [
                    {"themes": {"$eq": "friendship, courage, good vs evil, coming of age"}},
                    {"themes": {"$eq": "heroism, friendship, good vs evil, power corruption"}},
                    {"themes": {"$eq": "survival, oppression, sacrifice, rebellion"}},
                ]
            }
        )

        print(f"Found {len(results['ids'][0])} books that match multiple themes:")
        for i, (doc_id, document, distance) in enumerate(zip(
            results['ids'][0], results["documents"][0], results["distances"][0]
        )):
            metadata = results['metadatas'][0][i]
            print(f"  {i+1}. {metadata['title']} ({doc_id}) - Distance: {distance:.4f}")
            print(f"     Document: {document[:100]}...")

    except Exception as error:
        print(f"Error in advanced search: {error}")

if __name__ == "__main__":
    main()