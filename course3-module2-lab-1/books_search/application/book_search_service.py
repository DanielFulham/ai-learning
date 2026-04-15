from application.interfaces.book_search_service_interface import BookSearchServiceInterface
from application.result_printer import print_query_results, print_get_results


class BookSearchService(BookSearchServiceInterface):

    def __init__(self, collection):
        self.collection = collection

    def search_by_similarity(self, query_text: str, n_results: int) -> None:
        print(f"\nSearch for '{query_text}':")
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        print(f"Query: {query_text}")
        print_query_results(results)

    def filter_by_genre(self, genres: list) -> None:
        print(f"\nFilter for {' or '.join(genres)} genre:")
        results = self.collection.get(
            where={"genre": {"$in": genres}}
        )
        print(f"Found {len(results['ids'])} books in {' or '.join(genres)} genre:")
        print_get_results(results, fields=["genre"])

    def filter_by_rating(self, min_rating: float) -> None:
        print(f"\nFilter for books with rating {min_rating} or higher:")
        results = self.collection.get(
            where={"rating": {"$gte": min_rating}}
        )
        print(f"Found {len(results['ids'])} books with rating {min_rating} or higher:")
        print_get_results(results, fields=["rating"])

    def search_combined(self, query_text: str, genre: str, min_rating: float) -> None:
        print(f"\nCombined search for highly-rated {genre} books:")
        results = self.collection.query(
            query_texts=[query_text],
            n_results=5,
            where={
                "$and": [
                    {"genre": genre},
                    {"rating": {"$gte": min_rating}}
                ]
            }
        )
        print(f"Query: {query_text} with genre filter '{genre}' and rating filter {min_rating} or higher")
        print_query_results(results, fields=["rating"])

    def filter_by_decade(self, start_year: int, end_year: int) -> None:
        print(f"\nFilter for books published in the {start_year}s:")
        results = self.collection.get(
            where={
                "$and": [
                    {"year": {"$gte": start_year}},
                    {"year": {"$lt": end_year}}
                ]
            }
        )
        count = len(results['ids'])
        print(f"Found {count} book{'s' if count != 1 else ''} published in the {start_year}s:")
        print_get_results(results, fields=["year"])

    def filter_by_page_count(self, target: int, tolerance: int) -> None:
        print(f"\nSearch for books with similar page counts:")
        results = self.collection.get(
            where={
                "$and": [
                    {"pages": {"$gte": target - tolerance}},
                    {"pages": {"$lte": target + tolerance}}
                ]
            }
        )
        print(f"Found {len(results['ids'])} books with page counts between {target - tolerance} and {target + tolerance}:")
        print_get_results(results, fields=["pages"])

    def search_by_themes(self, query_text: str, themes: list) -> None:
        print(f"\nSearch for books that match multiple themes:")
        results = self.collection.query(
            query_texts=[query_text],
            n_results=5,
            where={"$or": [{"themes": {"$eq": theme}} for theme in themes]}
        )
        print(f"Found {len(results['ids'][0])} books that match multiple themes:")
        print_query_results(results)