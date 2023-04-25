import os
from fastapi import FastAPI, Depends, HTTPException, UploadFile, APIRouter, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
import crud, models, schemas
from database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

from typing import Dict

from pygpt4all.models.gpt4all_j import GPT4All_J
from completions.llm import (
    gen_index_from_doc,
    generate,
    load_index_in_memory,
    search_index,
    build_prompt,
    load_model
)

app = FastAPI()

DEBUG = True

MODEL = load_model("./model/ggml-gpt4all-j-v1.3-groovy.bin")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/create_document/", response_model=schemas.Document)
def create_document(document: schemas.Document, db: Session = Depends(get_db)):
    db_document = crud.get_document(db, uid=document.uid)
    if db_document:
        raise HTTPException(status_code=400, detail="Document with that UID already exists")
    return crud.create_document(db, document)

@app.get("/documents/", response_model=list[schemas.Document])
def read_documents(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    documents = crud.get_documents(db, skip, limit)
    return documents

@app.get("/documents/{uid}", response_model=schemas.Document)
def read_document(uid: str, db: Session = Depends(get_db)):
    db_document = crud.get_document(db, uid)
    if db_document is None:
        raise HTTPException(status_code=404, detail="Document with that UID does not exist")
    return db_document
    

@app.get("/healthcheck", include_in_schema=False)
def healthcheck() ->  Dict[str, str]:
    return {'status':'ok'}


@app.post("/create_index/", status_code=status.HTTP_201_CREATED)
async def create_index(file: UploadFile):
    uid, filename = gen_index_from_doc(file, num_pages=10 if DEBUG else None)
    return {
        "uid":uid,
        "filename":filename
    }

class CompletionRequest(BaseModel):
    query: str
    uid: str

@app.post("/get_completion/", status_code=status.HTTP_200_OK)
async def get_completion(completionRequest: CompletionRequest):
    index, chunks = load_index_in_memory(completionRequest.uid)
    context = search_index(chunks, index, completionRequest.query)
    prompt = build_prompt(
        context, completionRequest.query
    )
    response = generate(prompt, MODEL)
    return {
        "agentResponse":response
    }