import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
import nltk

# Download the Punkt tokenizer for NLTK
nltk.download("punkt_tab")

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
    "In 2023, I visited Paris.",
    "In 2022, I visited New York.",
    "In 2021, I visited New Orleans.",
    "Albert Einstein developed the theory of relativity.",
    "The Great Wall of China is one of the wonders of the world.",
    "Python is a popular programming language for data science.",
    "Mount Everest is the highest mountain in the world.",
    "The capital of France is Paris.",
    "In 1969, humans first landed on the moon.",
    "The Amazon Rainforest is the largest tropical rainforest on Earth.",
    "The COVID-19 pandemic began in 2019.",
    "The Pyramids of Giza were built over 4,500 years ago.",
    "Leonardo da Vinci painted the Mona Lisa in the early 1500s.",
    "The Internet was invented in the late 20th century.",
    "Shakespeare wrote 'Romeo and Juliet' in the 16th century.",
    "The first computer was developed in the 1940s.",
    "The Eiffel Tower was completed in 1889.",
    "The human genome project was completed in 2003.",
    "The first modern Olympic Games were held in Athens in 1896.",
    "The Sahara Desert is the largest hot desert in the world.",
    "The theory of evolution was proposed by Charles Darwin.",
    "Beethoven composed his Ninth Symphony in 1824.",
    "The United Nations was founded in 1945.",
    "Electric cars are becoming increasingly popular around the world.",
    "The human brain contains approximately 86 billion neurons.",
    "Mars is known as the Red Planet.",
    "The Great Barrier Reef is the world's largest coral reef system.",
    "Artificial intelligence is transforming various industries.",
    "The Renaissance was a period of great cultural revival in Europe.",
    "The Amazon River is the second-longest river in the world.",
    "Marie Curie was the first woman to win a Nobel Prize.",
    "The Titanic sank on its maiden voyage in 1912.",
    "Kangaroos are native to Australia.",
    "Mount Kilimanjaro is the tallest mountain in Africa.",
    "The Berlin Wall fell in 1989, marking the end of the Cold War.",
    "The first man-made satellite, Sputnik 1, was launched in 1957.",
    "Hubble Space Telescope has provided stunning images of the universe.",
    "The Nobel Prize is awarded annually for outstanding contributions in various fields.",
    "The Industrial Revolution began in the late 18th century.",
    "Blockchain technology is the foundation of cryptocurrencies.",
    "The Alhambra is a palace and fortress complex in Granada, Spain.",
    "The theory of general relativity revolutionized our understanding of gravity.",
    "The human body has 206 bones.",
    "The United States declared independence from Britain in 1776.",
    "The Taj Mahal was built in memory of Mumtaz Mahal.",
    "The first female prime minister of the UK was Margaret Thatcher.",
    "Vincent van Gogh painted 'Starry Night' in 1889.",
    "The ancient city of Petra in Jordan is known for its rock-cut architecture.",
    "The first smartphone was released in 1992.",
    "The moon's gravitational pull affects the Earth's tides."
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