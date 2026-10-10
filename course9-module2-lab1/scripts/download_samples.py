import urllib.request

from config import DOCUMENTS_DIR, SAMPLE_FILE_URLS


def download_sample_file(filename: str, url: str) -> None:
    target = DOCUMENTS_DIR / filename
    if target.exists() and target.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=10) as response:
        content = response.read()
    target.write_bytes(content)


def prepare_sample_files() -> None:
    """Create the documents directory and download the sample files into it."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in SAMPLE_FILE_URLS.items():
        download_sample_file(filename, url)


def main() -> None:
    prepare_sample_files()
    print(f"Sample files are in {DOCUMENTS_DIR}/")


if __name__ == "__main__":
    main()
