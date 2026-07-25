from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from brain.assistant import process_message
from memory.database import init_db


app = FastAPI()

init_db()


app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():

    return {"status": "running"}


@app.post("/chat")
def chat(data: ChatRequest):

    response = process_message(data.message)

    return {

        "response": response
    }