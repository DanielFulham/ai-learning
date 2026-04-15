from typing import Protocol


class BookSearchServiceInterface(Protocol):

    def search_by_similarity(self, query_text: str, n_results: int) -> None:
        ...

    def filter_by_genre(self, genres: list) -> None:
        ...

    def filter_by_rating(self, min_rating: float) -> None:
        ...

    def search_combined(self, query_text: str, genre: str, min_rating: float) -> None:
        ...

    def filter_by_decade(self, start_year: int, end_year: int) -> None:
        ...

    def filter_by_page_count(self, target: int, tolerance: int) -> None:
        ...

    def search_by_themes(self, query_text: str, themes: list) -> None:
        ...