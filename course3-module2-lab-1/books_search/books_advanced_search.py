from application.book_loader import load_books
from application.interfaces.book_search_service_interface import BookSearchServiceInterface
from application.container import build_search_service, build_client, build_embedding, build_repository
from interfaces.book_repository_interface import BookRepositoryInterface

collection_name = "book_collection"

def main():
    try:
        ef = build_embedding()
        client = build_client()
        books = load_books()

        repository: BookRepositoryInterface = build_repository(client)
        
        collection = repository.create_collection(collection_name, ef)

        print(f"Collection created: {collection.name}")
        
        repository.add_books(collection, books)

        all_items = collection.get()

        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")

        service: BookSearchServiceInterface = build_search_service(collection)

        print("=== Similarity Search Examples ===")
        service.search_by_similarity("magical fantasy adventure", n_results=3)
        service.filter_by_genre(["Fantasy", "Science Fiction"])
        service.filter_by_rating(4.5)
        service.search_combined(
            "dystopian society with themes of surveillance and freedom",
            genre="Dystopian",
            min_rating=4.2
        )
        service.filter_by_decade(1950, 1960)
        service.filter_by_page_count(target=300, tolerance=100)
        service.search_by_themes(
            "books that explore themes of friendship, courage, and good vs evil",
            themes=[
                "friendship, courage, good vs evil, coming of age",
                "heroism, friendship, good vs evil, power corruption",
                "survival, oppression, sacrifice, rebellion"
            ]
        )

    except Exception as error:
        print(f"Error: {error}")

if __name__ == "__main__":
    main()