from sqlalchemy.orm import Session

import models, schemas

def get_document(db: Session, uid: str):
    return db.query(
        models.Document
    ).filter(models.Document == uid).first()

def get_documents(db: Session, skip: int = 0, limit: int = 10):
    return db.query(models.Document).offset(skip).limit(limit).all()

def create_document(db: Session, document: schemas.Document):
    db_document = models.Document(uid=document.uid, name=document.name)
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document