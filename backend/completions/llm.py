import os
import uuid
from torch import Tensor
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from pygpt4all.models.gpt4all_j import GPT4All_J

from langchain.text_splitter import CharacterTextSplitter
from completions.utils import load_pdf, save_json, load_json

from fastapi import UploadFile
from typing import List, Union, Tuple, Optional

CACHE_PATH = os.path.abspath("../backend/cache/")
print(CACHE_PATH)

EMBEDDING_MODEL = SentenceTransformer("msmarco-distilbert-base-tas-b")

PROMPT_TEMPLATE = """The prompt below contains three sources as context and a question to be answered using the context. Write a response to the question using the context.
### Prompt:
Context:
{context}

Question:
{question}
### Response:"""

def load_model(path_or_name: str, remote: bool = False, **kwargs) -> Union[GPT4All_J, Tuple[AutoModelForCausalLM, AutoTokenizer]]:
    """
    Load model into memory.
    """
    if remote:
        tokenizer = AutoTokenizer.from_pretrained(path_or_name)
        model = AutoModelForCausalLM.from_pretrained(path_or_name, **kwargs)
        return model, tokenizer
    else:
        model = GPT4All_J(path_or_name)
        return model

def build_prompt(context: str, query: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, question=query)

# TODO: streaming w/ new_text_callback
def generate(prompt: str, model: Union[GPT4All_J, AutoModelForCausalLM], tokenizer: Optional[AutoTokenizer] = None, max_tokens: int = 100) -> str:
    if isinstance(model, GPT4All_J):
        output = model.generate(prompt, n_predict=max_tokens)
    else:
        if tokenizer is None:
            raise Exception("Must pass Tokenizer if using Transformers model")
        else:
            inputs = tokenizer(prompt, return_tensors="pt").input_ids
            outputs = model.generate(inputs, max_new_tokens=max_tokens, do_sample=True, top_p=0.9, temperature=0.9)
            output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return output

def build_index(chunks: List[str], embedding_func: callable, dims: int = 768) -> np.ndarray:
    index = np.zeros(
        (len(chunks), dims), dtype=np.float32
    )
    for i,chunk in enumerate(chunks):
        chunk_embedding = embedding_func(chunk)
        index[i] = chunk_embedding
    return index

def build_context(ranked_results: List[str]) -> str:
    cleaned = ["- "+r.replace("\n", " ") for r in ranked_results]
    return "\n\n".join(cleaned)

def gen_embedding(text: str, model: SentenceTransformer = EMBEDDING_MODEL) -> Union[List[Tensor], np.ndarray, Tensor]:
    text = text.replace("\n", " ")
    return model.encode(text)

def save_index(index: np.ndarray, raw_chunks: List[str], base_path: str = CACHE_PATH) -> str:
    out_name_index = uuid.uuid4().hex
    save_path_index = os.path.join(base_path, out_name_index)
    out_name_json = out_name_index+".json"
    np.save(save_path_index, index)
    save_json(raw_chunks, os.path.join(base_path, out_name_json))
    return out_name_index

def gen_index_from_doc(pdf_file: UploadFile, num_pages: Optional[int] = None, chunksize: int = 1000, overlap: int = 100) -> Tuple[np.ndarray, str]:
    pages = load_pdf(pdf_file.file, num_pages=num_pages)
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunksize,
        chunk_overlap=overlap,
        length_function=len
    )

    chunks = []
    for page in pages:
        chunks+=splitter.split_text(page)
    
    index = build_index(
        chunks, gen_embedding, 768
    )
    uid = save_index(index, chunks)
    return uid, pdf_file.filename
    
def load_index_in_memory(uid: str, base_path: str = CACHE_PATH) -> Tuple[np.ndarray, List[str]]:
    index_name = uid+".npy"
    json_name = uid+".json"
    index = np.load(os.path.join(base_path, index_name))
    chunks = load_json(os.path.join(base_path, json_name))
    return index, chunks

def search_index(chunks: List[str], index: np.ndarray, query: str, embedding_func: callable = gen_embedding) -> str:
    query_embedding = embedding_func(query)
    reranked = util.semantic_search(query_embedding, index, top_k=3)
    out = [
        chunks[int(r['corpus_id'])] for r in reranked[0]
    ]
    formatted_context = build_context(out)
    return formatted_context