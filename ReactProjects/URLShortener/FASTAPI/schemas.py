from pydantic import BaseModel

class URLCreate(BaseModel):
    alias : str
    original_url : str

class URLResponse(BaseModel):
    alias : str
    original_url: str
    clicks : int

class Config:
    from_attributes = True