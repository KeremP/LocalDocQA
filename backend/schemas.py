from pydantic import BaseModel

class DocumentBase(BaseModel):
    uid: str
    name: str

class Document(DocumentBase):
    id: int

    class Config:
        orm_mode = True