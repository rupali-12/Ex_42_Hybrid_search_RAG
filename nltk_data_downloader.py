import nltk
import os

class NLTKDataDownloader:
    def __init__(self, download_dir):
        self.download_dir = download_dir
        self._create_download_dir()

    def _create_download_dir(self):
        """Create the download directory if it doesn't exist."""
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def download_data(self, resources):
        """Download NLTK resources specified in the resources list."""
        for resource in resources:
            try:
                nltk.data.find(resource)
                print(f"'{resource}' already downloaded.")
            except LookupError:
                print(f"Downloading '{resource}'...")
                nltk.download(resource, download_dir=self.download_dir)

# Example usage
if __name__ == "__main__":
    downloader = NLTKDataDownloader(download_dir="nltk_data")
    downloader.download_data(["tokenizers/punkt", "corpora/stopwords"])
