from fastapi import FastAPI
from starlette.requests import Request
from rag import RagGraph


app = FastAPI()
rag_app = RagGraph()


@app.get("/health_check")
def health_check():
    response = {"message": "I AM OK!"}
    return response


@app.get("/chat")
def chat(request: Request):
    query = request.query_params["query"]
    result = rag_app.invoke(query)

    chat_history = rag_app.chat_history
    for i, msg in enumerate(chat_history):
        if i % 2 == 0:
            chat_history[i] = f"User: {msg}"
        else:
            chat_history[i] = f"AI: {msg}"

    response = {"query": query, "response": result['answer'], "chat_history": chat_history}
    return response


@app.get("/clear_history")
def clear_history():
    rag_app.chat_history = []
    response = {"message": "Chat History cleared!", "chat_history": rag_app.chat_history}
    return response
