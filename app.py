import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
import os
from dotenv import load_dotenv

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
    ]

    bm25_encoder.fit(sentences)

    # Optionally save and load encoder values
    bm25_encoder.dump("bm25_values.json")
    bm25_encoder = BM25Encoder().load("bm25_values.json")

    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
    )

    retriever.add_texts(sentences)

    # User input for query
    query = st.text_input("Enter your query:", "What city did I visit first?")

    if query:
        # Perform the hybrid search
        results = retriever.invoke(query)
        st.write("Search Results:")
        st.write(results)

