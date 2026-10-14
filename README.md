# Demo-2

A tiny FastAPI app.

## Run it

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open:

- http://127.0.0.1:8000: the app
- http://127.0.0.1:8000/docs: interactive API docs

## Endpoints

| Method | Path            | Response                          |
| ------ | --------------- | --------------------------------- |
| GET    | `/`             | `{"message": "Hello, World!"}`    |
| GET    | `/hello/{name}` | `{"message": "Hello, {name}!"}`   |
| GET    | `/health`       | `{"status": "ok"}`                |
