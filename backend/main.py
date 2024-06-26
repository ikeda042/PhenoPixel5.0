from fastapi import FastAPI
import uvicorn
from tmp import init


app = FastAPI()


@app.get("/")
async def read_root():
    return {"Response": "Root"}


@app.get("/init")
async def init():
    init("sk328gen120min.tif", mode="dual")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
