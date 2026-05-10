from fastapi import FastAPI
from fastapi import HTTPException, Depends
from fastapi.responses import RedirectResponse
from database import Base
from database import engine
from database import SessionLocal
from schemas import URLCreate, URLResponse
from models import URL
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


Base.metadata.create_all(bind = engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
    


@app.post("/shorten", response_model= URLResponse)
def shorten_url(url_data: URLCreate, db : Session = Depends(get_db)):
    if db.query(URL).filter(URL.alias == url_data.alias).first() is not None:
        raise HTTPException(
            status_code=400,
            detail=f"Alias {url_data.alias} found, Unsuccessful Shorten"
        )
    
    new_url = URL(alias = url_data.alias, original_url = url_data.original_url)
    
    db.add(new_url)
    db.commit()
    db.refresh(new_url)
    return new_url

@app.get("/{code}")
def get_url(code: str, db : Session = Depends(get_db)):
    old_url = db.query(URL).filter(URL.alias == code).first()
    if old_url is None:
        raise HTTPException(
            status_code=404,
            detail=f"{code} not found"
        )
    old_url.clicks += 1
    db.commit()
    return RedirectResponse(old_url.original_url, 302)

@app.get("/{code}/stats", response_model= URLResponse)
def check_clicks(code : str,db : Session = Depends(get_db)):
    requested_url = db.query(URL).filter(URL.alias == code).first()
    if requested_url is None:
        raise HTTPException(
            status_code=404,
            detail=f"{code} not found"
        )
    return(requested_url)
    
@app.delete("/{code}/remove_alias",response_model= str)
def remove_alias(code : str, db : Session = Depends(get_db)):
    requested_alias = db.query(URL).filter(URL.alias == code).first()
    if requested_alias is None:
        raise HTTPException(
            status_code= 404,
            detail = f"{code} not found"
        )
    old_url = requested_alias.original_url
    db.delete(requested_alias)
    db.commit()
    return old_url
    