import os
import logging

import google.generativeai as genai

import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cohere

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]-%(message)s"
)
logger = logging.getLogger(__name__)

try:
    genai.configure(api_key=os.getenv('api_key'))
    qdrant = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'),timeout=60)
    gemini = genai.GenerativeModel('gemini-1.5-flash')
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
    collection_name = 'rag_docs'
    if not  qdrant.collection_exists(collection_name):
        qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )
        logger.info("qdrant collection created succesfully")
        print('collections created succesfully')
    else:
        logger.info('collectin already exists so skipping creation')
        print('collections already exist ')
except Exception as e:
    logger.error(f"error in setup {e}")
    raise


def read_pdf_return_emb(pdf_path):
    try:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = []
        for doc in docs:
            splits = text_splitter.split_text(doc.page_content)
            for split in splits:
                chunks.append(split)
        points = []
        for i, chunk in enumerate(chunks):
            embed = embedder.encode(chunk).tolist()
            points.append(models.PointStruct(
                id=i,
                vector=embed,
                payload={'text': chunk}
            ))
        qdrant.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info("pdf embedded and stored in qdrant")
    except Exception as e:
        logger.error(f"Error reading or embedding the pdf {e}")
        raise


def retrieve(query, top_k=10):
    try:
        query_vec = embedder.encode(query).tolist()
        results = qdrant.search(
            collection_name=collection_name,
            query_vector=query_vec,
            limit=top_k
        )
        logger.info(f"Retrieved {len(results)} results for query: {query}")
        return results
    except Exception as e:
        logger.error(f"Error in retrieval: {e}")
        return []


def answer_query(query):
    try:
        retrieved = retrieve(query, top_k=10)
        if not retrieved:
            return "No relevant results found", []
        docs = [r.payload['text'] for r in retrieved]
        rerank_results = co.rerank(
            query=query,
            documents=docs,
            top_n=3,
            model="rerank-english-v3.0"
        )
        reranked_docs = [docs[r.index] for r in rerank_results.results]
        context = "\n".join(reranked_docs)
        prompt = f"""
        Use the following context to answer the query.
        Context:
        {context}
        Query: {query}
        Answer with citations like [1], [2].
        """
        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
        logger.info("Generated answer successfully")
        return response.text, reranked_docs
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Error generating answer", []


st.title("RAG with Gemini + Qdrant")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    read_pdf_return_emb("temp.pdf")
    st.success("PDF uploaded and processed")

query = st.text_input("Ask a question")
if st.button("Submit Query") and query:
    answer, retrieved_docs = answer_query(query)
    st.write(" Answer")
    st.write(answer)
    if retrieved_docs:
        st.write("Sources")
        for i, doc in enumerate(retrieved_docs, 1):
            st.write(f"[{i}] {doc[:200]}...")
