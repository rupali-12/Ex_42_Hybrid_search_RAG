import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
import nltk

# Download the Punkt tokenizer for NLTK
nltk.download("punkt")

# Load environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

# Function to initialize Pinecone and create an index if it doesn't exist
def initialize_pinecone(index_name):
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # dimensionality of dense model
            metric="dotproduct",  # sparse values supported only for dotproduct
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)

# Function to initialize embeddings and BM25 encoder
def initialize_embeddings_and_encoder():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    bm25_encoder = BM25Encoder().default()
    return embeddings, bm25_encoder

# Function to fit and save BM25 encoder
def fit_and_save_bm25_encoder(bm25_encoder, sentences, file_name="bm25_values.json"):
    bm25_encoder.fit(sentences)
    bm25_encoder.dump(file_name)

# Function to load BM25 encoder from a file
def load_bm25_encoder(file_name="bm25_values.json"):
    return BM25Encoder().load(file_name)

# Initialize the app components
index_name = "hybrid-search-langchain-pinecone"
index = initialize_pinecone(index_name)
embeddings, bm25_encoder = initialize_embeddings_and_encoder()

# Sentences to encode
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

# Fit and save the BM25 encoder only if the file doesn't exist
if not os.path.exists("bm25_values.json"):
    fit_and_save_bm25_encoder(bm25_encoder, sentences)

# Load the BM25 encoder
bm25_encoder = load_bm25_encoder()

# Initialize the retriever
retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
)

# Add texts to the retriever
retriever.add_texts(sentences)

# Streamlit UI
st.title("🔍 Hybrid Search with Pinecone and BM25")
st.write("This application uses hybrid search with Pinecone and BM25 encoding. Enter a query to search through the stored sentences.")

# User input
query = st.text_input("Enter your query:", placeholder="What city did I visit recently?")

# Search and display the results
if st.button("Search"):
    if query.strip():
        try:
            result = retriever.invoke(query)
            if result:
                st.subheader("Search Results:")
                st.write(result)
            else:
                st.write("No matching results found.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a query to search.")