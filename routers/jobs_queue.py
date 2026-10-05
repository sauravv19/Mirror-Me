import os
import uuid
import json
import base64
import time
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import Optional, Literal, List

import requests
from fastapi import (
    APIRouter,
    File,
    UploadFile,
    Form,
    Depends,
    HTTPException,
    status,
)
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers, UploadFile as StarletteUploadFile
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReturnDocument
from google.auth.crypt import RSASigner

from routers.tryon_pose import tryon_request
from routers.auth import get_current_user

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URI)
db = client["fashion_app"]
jobs_coll = db["jobs"]

with open(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")) as f:
    service_account_info = json.load(f)

BUCKET_NAME = "ai-fashion-app-mirror-me"

def generate_signed_url(
    bucket_name: str,
    object_name: str,
    service_account_info: dict,
    expiration: int = 15 * 60,
    content_type: str = "application/octet-stream"
) -> str:
    method = "PUT"
    expires_at = int(time.time() + expiration)
    canonical_uri = f"/{bucket_name}/{object_name}"
    string_to_sign = "\n".join([
        method,
        "", 
        content_type,
        str(expires_at),
        canonical_uri
    ])
    signer = RSASigner.from_string(service_account_info["private_key"])
    signature = base64.b64encode(signer.sign(string_to_sign.encode())).decode()
    client_email = service_account_info["client_email"]

    return (
        f"https://storage.googleapis.com{canonical_uri}"
        f"?GoogleAccessId={client_email}"
        f"&Expires={expires_at}"
        f"&Signature={requests.utils.quote(signature)}"
    )

class DummyUploadFile(StarletteUploadFile):
    def __init__(self, path: str):
        f = open(path, "rb")
        filename = os.path.basename(path)
        headers = Headers({"content-type": "image/png"})
        super().__init__(filename=filename, file=f, headers=headers)

class JobUpdate(BaseModel):
    status: Optional[Literal["waiting","running","failed","completed"]]
    percentage: Optional[int] = Field(None, ge=0, le=100)
    output_path: Optional[str]

router = APIRouter(
     tags=["Jobs"], 
)

@router.post("/create_job", summary="Enqueue a new try‑on job", dependencies=[Depends(get_current_user)])
async def create_job(
    person_image: UploadFile = File(...),
    overlay_cloths: List[UploadFile] = File(...),
    overlay_model_types: str = Form(...),
    overlay_garment_types: str = Form(...),
    overlay_styles: str = Form(...),
    opacity: str = Form(..., description="Enter how opaque you want the cloth to be"),
    fabric: str = Form(..., description="Enter the fabric of the cloth"),    
    instructions: str = Form(""),
    current_user: dict = Depends(get_current_user),
):
    job_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    user_id = str(current_user["_id"])

    ext = os.path.splitext(person_image.filename)[1]
    person_gcs = f"input/{job_id}/person{ext}"
    person_url = generate_signed_url(BUCKET_NAME, person_gcs, service_account_info, content_type="application/octet-stream")
    data = await person_image.read()
    r = requests.put(person_url, data=data, headers={"Content-Type": "application/octet-stream"})
    if not r.ok:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Person upload failed: {r.text}")
    person_public = f"https://storage.googleapis.com/{BUCKET_NAME}/{person_gcs}"

    cloth_urls = []
    for idx, cloth in enumerate(overlay_cloths):
        ext = os.path.splitext(cloth.filename)[1]
        cloth_gcs = f"input/{job_id}/cloth_{idx}{ext}"
        upload_url = generate_signed_url(BUCKET_NAME, cloth_gcs, service_account_info, content_type="application/octet-stream")
        data = await cloth.read()
        r2 = requests.put(upload_url, data=data, headers={"Content-Type": "application/octet-stream"})
        if not r2.ok:
            raise HTTPException(status_code=500, detail=f"Cloth #{idx} upload failed: {r2.text}")
        cloth_urls.append(f"https://storage.googleapis.com/{BUCKET_NAME}/{cloth_gcs}")

    model_types = [m.strip() for m in overlay_model_types.split(",")]
    garment_types = [g.strip() for g in overlay_garment_types.split(",")]
    styles = [s.strip() for s in overlay_styles.split(",")]

    job_doc = {
        "job_id": job_id,
        "user_id": user_id,
        "status": "waiting",
        "percentage": 0,
        "created_at": created_at,
        "started_at": None,
        "completed_at": None,
        "input_paths": {
            "person": person_public,
            "cloths": cloth_urls
        },
        "overlay_model_types": model_types,
        "overlay_garment_types": garment_types,
        "overlay_styles": styles,
        "opacity": opacity,
        "fabric": fabric,
        "instructions": instructions,
        "output_path": None,
        "gemini_output": None,
    }
    await jobs_coll.insert_one(job_doc)

    return {"job_id": job_id}


@router.get("/jobs/next", summary="Worker: fetch & lock next waiting job")
async def get_next_job():
    """
    This endpoint is global (no user scoping). It picks the oldest 'waiting' job,
    marks it 'running', and returns its payload.
    """
    now = datetime.now(timezone.utc)
    job = await jobs_coll.find_one_and_update(
        {"status": "waiting"},
        {"$set": {"status": "running", "percentage": 0, "started_at": now}},
        sort=[("created_at", 1)],
        projection={"_id": 0},
        return_document=ReturnDocument.AFTER
    )
    if not job:
        return {"message": "No waiting jobs."}

    try:
        final_path, gemini_path = await _run_and_upload(job)
        await jobs_coll.update_one(
            {"job_id": job["job_id"]},
            {"$set": {
                "status": "completed",
                "percentage": 100,
                "completed_at": datetime.now(timezone.utc),
                "output_path": final_path,
                "gemini_output": gemini_path,
            }}
        )
        return {"job_id": job["job_id"], "status": "completed"}
    except Exception as e:
        await jobs_coll.update_one(
            {"job_id": job["job_id"]},
            {"$set": {"status": "failed", "percentage": 0}}
        )
        raise HTTPException(status_code=500, detail=f"Job failed: {e}")


@router.get("/jobs/{job_id}", summary="Get a single job by job_id", dependencies=[Depends(get_current_user)])
async def get_job(job_id: str, current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    job = await jobs_coll.find_one({"job_id": job_id, "user_id": user_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/get_all_jobs", summary="Fetch all jobs for current user", dependencies=[Depends(get_current_user)])
async def get_all_jobs(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    cursor = jobs_coll.find({"user_id": user_id}, {"_id": 0}).sort("created_at", 1)
    return await cursor.to_list(length=1000)


@router.delete("/delete_jobs", summary="Delete all jobs for current user", dependencies=[Depends(get_current_user)])
async def delete_all_jobs(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["_id"])
    await jobs_coll.delete_many({"user_id": user_id})
    return {"deleted": True}

async def _run_and_upload(job: dict) -> str:
    """
    Downloads inputs, runs `tryon_request`, uploads result,
    and returns the public GCS URL of the final image.
    """
    with tempfile.TemporaryDirectory() as tmp:
        parsed = urlparse(job["input_paths"]["person"])
        pfile = os.path.join(tmp, os.path.basename(parsed.path))
        resp = requests.get(job["input_paths"]["person"])
        resp.raise_for_status()
        with open(pfile, "wb") as f:
            f.write(resp.content)

        cloth_paths = []
        for url in job["input_paths"]["cloths"]:
            parsed = urlparse(url)
            cfile = os.path.join(tmp, os.path.basename(parsed.path))
            r = requests.get(url)
            r.raise_for_status()
            with open(cfile, "wb") as f:
                f.write(r.content)
            cloth_paths.append(cfile)

        person_up = DummyUploadFile(pfile)
        cloth_ups = [DummyUploadFile(p) for p in cloth_paths]

        out_path, gemini_output = await tryon_request(
            person_up,
            cloth_ups,
            ",".join(job["overlay_model_types"]),
            ",".join(job["overlay_garment_types"]),
            ",".join(job["overlay_styles"]),
            ",".join(job["opacity"]),
            ",".join(job["fabric"]),
            job.get("instructions", "")
        )

        gcs_out = f"output/{job['job_id']}/result.png"
        upload_url = generate_signed_url(BUCKET_NAME, gcs_out, service_account_info, content_type="image/png")
        with open(out_path, "rb") as f:
            put_r = requests.put(upload_url, data=f.read(), headers={"Content-Type": "image/png"})
        put_r.raise_for_status()

        return f"https://storage.googleapis.com/{BUCKET_NAME}/{gcs_out}", gemini_output
