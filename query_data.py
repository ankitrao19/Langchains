from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import shutil
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import Optional


app = FastAPI()

CHROMA_PATH = "/home/ankit/langchain/chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


class QueryRequest(BaseModel):
    query: str
    mapping_criteria: float

def process_file(file: UploadFile = File(...), file_name: str = Form(...), api_key: str = Form(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
        documents = load_documents(temp_file.name)
        chunks = split_text(documents)
        save_to_chroma(chunks, api_key)

        return JSONResponse(
            status_code=200, content={"message": "DB has been created successfully."}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_documents(file_path: str):
    loader = PyPDFLoader(file_path)  
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def save_to_chroma(chunks: list[Document], api_key):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(
            openai_api_key= api_key
        ),
        persist_directory=CHROMA_PATH,
    )
    db.persist()


def get_formatted_response(query_text: str, accuracy:float, api_key:str) -> str:
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < accuracy:
        raise HTTPException(status_code=404, detail="Unable to find matching results.")
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = ChatOpenAI()
    response_text = model.predict(prompt)
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response

@app.post("/")
def welcome():
    return {"response": "Welcome to the FastAPI application!"}

@app.post("/query")
def process_query(request: Request, query_request: QueryRequest):
    api_key = request.headers.get("x-api-key")
    os.environ['OPENAI_API_KEY'] = api_key
    query_text = query_request.query
    accuracy = query_request.mapping_criteria
    formatted_response = get_formatted_response(query_text, accuracy, api_key)
    return {"formatted_response": formatted_response}

@app.post("/process_file")
async def upload_file(file: UploadFile = File(...), file_name: str = Form(...), x_api_key: Optional[str] = Header(None)):
    api_key=x_api_key
    print(api_key)
    return process_file(file=file, file_name=file_name , api_key=api_key)
