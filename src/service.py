import uvicorn
from main import app


if __name__ == "__main__":
    uvicorn.run(app, timeout_keep_alive=60)
