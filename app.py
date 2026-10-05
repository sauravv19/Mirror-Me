
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import warnings
import uvicorn
import motor.motor_asyncio
import time
import os
import threading
import logging
from routers import tryon_pose, jobs_queue, size, auth

warnings.filterwarnings("ignore")

app = FastAPI()
logging.basicConfig(
    filename="job_processor.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client["fashion_app"]
jobs_coll = db["jobs"]

async def fetch_and_queue_job():
    from pymongo import ReturnDocument
    from datetime import datetime, timezone

    job = await jobs_coll.find_one_and_update(
        {"status": "waiting"},
        {"$set": {"status": "queued", "queued_at": datetime.now(timezone.utc)}},
        sort=[("created_at", 1)],
        return_document=ReturnDocument.AFTER
    )
    return job

def worker_run_jobs():
    import requests
    while True:
        try:
            res = requests.get(f"http://localhost:8000/jobs/next")
            print(res)
            response = res.json()
            print(response)
            logger.info(f"Job result: {response}")
        except Exception as e:
            logger.error(f"Error processing job: {e}")
        time.sleep(20)
        
@app.on_event("startup")
def start_workers():
    t1 = threading.Thread(target=worker_run_jobs, daemon=True)
    t2 = threading.Thread(target=worker_run_jobs, daemon=True)
    t1.start()
    t2.start()
    logger.info("Background workers started.")
    print("Background workers started.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tryon_pose.router)
app.include_router(jobs_queue.router)
app.include_router(size.router)
app.include_router(auth.router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
