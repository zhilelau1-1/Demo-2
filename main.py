from fastapi import FastAPI

app = FastAPI(title="Tiny FastAPI App")


@app.get("/")
def root():
    return {"message": "Hello, World!"}


@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"Hello, {name}!"}


@app.get("/health")
def health():
    return {"status": "ok"}
