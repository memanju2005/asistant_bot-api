from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import uvicorn
import os

# LangChain Imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import Cohere

# --- Step 1: Load all documents ---
file_paths = ["data/resume_data.txt", "data/portfolio_data.txt", "data/mkcrack.txt"]
documents = []

for path in file_paths:
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    documents.extend(docs)

# --- Step 2: Chunk the documents ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# --- Step 3: Embeddings using HuggingFace API ---
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- Step 4: Create FAISS vector store ---
vectorstore = FAISS.from_documents(chunks, embeddings)

# --- Step 5: Initialize Cohere LLM ---
llm = Cohere(
    cohere_api_key=os.environ["COHERE_API_KEY"],
    model="command"
)

# --- Step 6: Custom Prompt Template ---
prompt_template = """You are Emo — Manjunath K’s personal portfolio assistant bot 🤖.
You're helpful, polite, chill, and only answer questions based on the provided context.

🧠 Your job is to strictly use the context to answer.
DO NOT explain or describe anything that is not already mentioned in the context.
DO NOT generate generic or external knowledge like Java definitions or technology overviews.

📋 Format your answers as a clean list of bullet points.

🙈 If the question is off-topic (e.g., celebrity news, science facts, or anything NOT about Manjunath’s portfolio),
respond with a light, funny message like:
"Oops! I'm just Emo, a humble portfolio bot. Ask me something about Manjunath’s skills, projects, or cool tech stuff!"

Context:
{context}

Question: {question}
Answer in bullet points :
"""

custom_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# --- Step 7: QA Chain Setup ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# --- FastAPI App Setup ---
app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your actual frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root Endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <h2>🤖 Portfolio Chatbot API is running!</h2>
    <p>Use <code>POST /ask</code> to query the bot.</p>
    """

# POST Request Model
class Query(BaseModel):
    query: str

# Chat Endpoint
@app.post("/ask")
async def ask_bot(query: Query):
    answer = qa_chain.run(query.query)
    return {"answer": answer}

# Uvicorn Entry Point (used in local dev, not in Render)
if __name__ == "__main__":
    uvicorn.run("chatbot_api:app", host="127.0.0.1", port=8000, reload=True)
