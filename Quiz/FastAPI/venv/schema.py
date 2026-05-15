from pydantic import BaseModel

class QuestionCreate(BaseModel):
    text : str

class QuestionResponse(BaseModel):
    id : int
    text : str
    class Config:
        from_attributes = True

class AnswerCreate(BaseModel):
    text : str
    question_id : int
    is_correct : bool

class AnswerResponse(BaseModel):
    id : int
    text : str
    question_id : int 
    is_correct : bool
    class Config:
        from_attributes = True

class ScoreCreate(BaseModel):
    player_id : int
    score : int

class ScoreResponse(BaseModel):
    id : int
    player_id : int
    score : int
    class Config:
        from_attributes = True

class UserCreate(BaseModel):
    name : str


class UserResponse(BaseModel):
    id : int
    name: str
    class Config:
        from_attributes = True







