from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import uvicorn
import os

# LangChain Imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Embedding and LLM
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import Cohere


# --- Load and Split Documents ---
file_paths = ["data/resume_data.txt", "data/portfolio_data.txt", "data/mkcrack.txt"]
documents = []
for path in file_paths:
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    documents.extend(docs)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# --- Embedding with HuggingFace API Token ---
HF_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Embed chunks manually (required for API-based embedding)
texts = [doc.page_content for doc in chunks]
metadatas = [doc.metadata for doc in chunks]

vectors = embeddings.embed_documents(texts)
# Convert to FAISS-compatible format
from langchain.docstore.document import Document
docs = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas)]

vectorstore = FAISS.from_embeddings(texts, vectors, metadatas)

# --- Cohere LLM ---
llm = Cohere(
    cohere_api_key=os.environ["COHERE_API_KEY"],
    model="command"
)

# --- Prompt Template ---
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

# --- QA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# --- FastAPI Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <h2>🤖 Portfolio Chatbot API is running!</h2>
    <p>Use <code>POST /ask</code> to query the bot.</p>
    """

class Query(BaseModel):
    query: str

@app.post("/ask")
async def ask_bot(query: Query):
    answer = qa_chain.run(query.query)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run("chatbot_api:app", host="127.0.0.1", port=8000, reload=True)
