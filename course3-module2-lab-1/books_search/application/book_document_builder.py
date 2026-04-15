def build_book_documents(books):
    return [book.to_document() for book in books]


def build_book_metadatas(books):
    return [book.to_metadata() for book in books]