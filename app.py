# from nltk_data_downloader import NLTKDataDownloader  # Import the downloader class
# import streamlit as st
# from pinecone import Pinecone, ServerlessSpec
# from langchain_community.retrievers import PineconeHybridSearchRetriever
# from langchain_huggingface import HuggingFaceEmbeddings
# from pinecone_text.sparse import BM25Encoder
# import os
# from dotenv import load_dotenv

# import nltk
# import os

# # Create a directory to store NLTK data if it doesn't exist
# nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
# if not os.path.exists(nltk_data_dir):
#     os.makedirs(nltk_data_dir)

# # Download the punkt tokenizer to the nltk_data directory
# try:
#     nltk.download('punkt', download_dir=nltk_data_dir)
# except Exception as e:
#     st.error(f"Error downloading NLTK data: {e}")


# # Set the NLTK data path to the local directory
# nltk.data.path.append(nltk_data_dir)


# # Load environment variables
# load_dotenv()

# # Streamlit UI
# st.title("Hybrid Search with Langchain and Pinecone")
# st.write("This app demonstrates hybrid search using Langchain and Pinecone.")

# # Pinecone API Key
# api_key = os.getenv("PINECONE_API_KEY")

# if not api_key:
#     st.error("Please set the Pinecone API key in the environment variables.")
# else:
#     # Initialize Pinecone client
#     pc = Pinecone(api_key=api_key)
#     index_name = "hybrid-search-langchain-pinecone"

#     # Create or retrieve the index
#     if index_name not in pc.list_indexes().names():
#         pc.create_index(
#             name=index_name,
#             dimension=384,
#             metric="dotproduct",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#         )

#     index = pc.Index(index_name)

#     # Load embeddings and BM25 encoder
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     bm25_encoder = BM25Encoder().default()

#     # Sample sentences for encoding
#     sentences = [
#         "In 2023, I visited Paris",
#         "In 2022, I visited New York",
#         "In 2021, I visited New Orleans",
#          "Albert Einstein developed the theory of relativity.",
#     "The Great Wall of China is one of the wonders of the world.",
#     "Python is a popular programming language for data science.",
#     "Mount Everest is the highest mountain in the world.",
#     "The capital of France is Paris.",
#     "In 1969, humans first landed on the moon.",
#     "The Amazon Rainforest is the largest tropical rainforest on Earth.",
#     "The COVID-19 pandemic began in 2019."
#     ]

#     bm25_encoder.fit(sentences)

#     # Optionally save and load encoder values
#     bm25_encoder.dump("bm25_values.json")
#     bm25_encoder = BM25Encoder().load("bm25_values.json")

#     retriever = PineconeHybridSearchRetriever(
#         embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
#     )

#     retriever.add_texts(sentences)

#     # # User input for query
#     query = st.text_input("Enter your query:", "What city did I visit first?")

#     if query:
#         # Perform the hybrid search
#         results = retriever.invoke(query)
#         st.write("Search Results:")
#         st.write(results)

# # queries = [
# #     "What city did I visit in 2023?",
# #     "Who developed the theory of relativity?",
# #     "What is the highest mountain in the world?",
# #     "What year did humans first land on the moon?",
# #     "What is the capital of France?",
# #     "When did the COVID-19 pandemic begin?"
# # ]

# # Perform hybrid search for each query and print the results
# # for query in queries:
# #     results = retriever.invoke(query)
# #     st.write(f"Query: {query}")
# #     st.write("Search Results:")
# #     st.write(results)




# **************************************
import os
from dotenv import load_dotenv
import streamlit as st
from nltk_data_downloader import NLTKDataDownloader  # Ensure this import matches your file structure
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder

# Initialize NLTK data downloader
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")  # Path for nltk_data
downloader = NLTKDataDownloader(download_dir=nltk_data_dir)
downloader.setup()  # Set up NLTK data path and download resources

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("Hybrid Search with Langchain and Pinecone")
st.write("This app demonstrates hybrid search using Langchain and Pinecone.")

# Pinecone API Key
api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    st.error("Please set the Pinecone API key in the environment variables.")
else:
    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)
    index_name = "hybrid-search-langchain-pinecone"

    # Create or retrieve the index
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(index_name)

    # Load embeddings and BM25 encoder
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    bm25_encoder = BM25Encoder().default()

    # Sample sentences for encoding
    sentences = [
        "In 2023, I visited Paris",
        "In 2022, I visited New York",
        "In 2021, I visited New Orleans",
        "Albert Einstein developed the theory of relativity.",
        "The Great Wall of China is one of the wonders of the world.",
        "Python is a popular programming language for data science.",
        "Mount Everest is the highest mountain in the world.",
        "The capital of France is Paris.",
        "In 1969, humans first landed on the moon.",
        "The Amazon Rainforest is the largest tropical rainforest on Earth.",
        "The COVID-19 pandemic began in 2019."
    ]

    bm25_encoder.fit(sentences)

    # Optionally save and load encoder values
    bm25_encoder.dump("bm25_values.json")
    bm25_encoder = BM25Encoder().load("bm25_values.json")

    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
    )

    retriever.add_texts(sentences)

    # User input for quer
    query = st.text_input("Enter your query:", "What city did I visit first?")

    if query:
        # Perform the hybrid search
        results = retriever.invoke(query)
        st.write("Search Results:")
        st.write(results)
