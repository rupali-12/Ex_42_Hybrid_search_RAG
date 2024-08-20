import os
import nltk
import streamlit as st

class NLTKDataDownloader:
    def __init__(self, download_dir="nltk_data"):
        self.download_dir = download_dir
        self.ensure_directory()

    def ensure_directory(self):
        """Ensure that the NLTK data directory exists."""
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def download_data(self, resources):
        """Download the specified NLTK resources if they are not already downloaded."""
        for resource in resources:
            try:
                nltk.download(resource, download_dir=self.download_dir)
                st.success(f"Downloaded {resource} successfully.")
            except Exception as e:
                st.error(f"Error downloading {resource}: {e}")

    def setup(self):
        """Setup NLTK data path and download necessary resources."""
        nltk.data.path.append(self.download_dir)
        self.download_data(["punkt", "stopwords"])
