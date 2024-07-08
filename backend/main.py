from fastapi import FastAPI


app = FastAPI()

cors_origins = [
    "localhost",
    "localhost:8080",
    "localhost:3000",
]


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}
