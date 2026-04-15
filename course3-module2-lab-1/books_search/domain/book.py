from dataclasses import dataclass

@dataclass
class Book:
    id: str
    title: str
    author: str
    genre: str
    year: int
    rating: float
    pages: int
    description: str
    themes: str
    setting: str

    def to_document(self) -> str:
        document = f"{self.title} with the description {self.description} explores themes of {self.themes} and is set in {self.setting}."
        document += f" It is a {self.genre} book published in {self.year}."
        document += f" Written by {self.author}."
        return document

    def to_metadata(self) -> dict:
        return {
            "title": self.title,
            "author": self.author,
            "genre": self.genre,
            "year": self.year,
            "rating": self.rating,
            "pages": self.pages,
            "description": self.description,
            "themes": self.themes,
            "setting": self.setting
        }