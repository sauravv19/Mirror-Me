import uuid
from fastapi import UploadFile, File, Form, HTTPException, Depends, status
from fastapi.routing import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from utils.tryon import multi_try_on
from pydantic import BaseModel, Field
from typing import Optional, Literal
import traceback
import requests
import uuid
import os
import time
import json
import base64
from PIL import Image
from google.auth.crypt import RSASigner
from dotenv import load_dotenv
from typing import Dict, Any, List
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient
from routers.auth import get_current_user
from google import genai
from google.genai import types


load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
class JobUpdate(BaseModel):
    status: Optional[Literal["waiting","running","failed","completed"]]
    percentage: Optional[int] = Field(None, ge=0, le=100)
    output_path: Optional[str]

MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
BUCKET_NAME = "ai-fashion-app-mirror-me"
db = client["fashion_app"]
jobs = db["jobs"]

url   = "https://kteoe10gd1dxpt-8000.proxy.runpod.net/pose_transfer"

router = APIRouter(dependencies=[Depends(get_current_user)])
with open(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")) as f:
    service_account_info = json.load(f)

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

@router.post("/face_swap")
async def face_swap(
    target_image: UploadFile = File(..., description="Source image with face to copy"),
    source_image: UploadFile = File(..., description="Source image with face to copy"),
):
    try:
        # Read image bytes
        source_bytes = await source_image.read()
        target_bytes = await target_image.read()
        
        # Prepare the face swap prompt
        prompt_text = """
Copy and paste the face from the second input image with absolutely no modification.
The face, skin texture, and expression must match the second image pixel-for-pixel, allowing only blending at boundaries with the garment.
Do not synthesize, redraw, or reinterpret any facial features of the second input.
Lock the entire body pose and limb locations to exactly match the first input. Absolutely no change in position, posture, scale, or silhouette is permitted
Extract and strictly follow a pose map or body landmarks from the first input throughout generation.
Never adjust the position, orientation, or posture of the person in the first image
"""

        # Prepare the multi-modal input parts
        parts = [
            types.Part(text=prompt_text),
            types.Part(inline_data=types.Blob(
                mime_type=source_image.content_type, 
                data=source_bytes
            )),
            types.Part(inline_data=types.Blob(
                mime_type=target_image.content_type,
                data=target_bytes
            ))
        ]

        # Generate the face swap
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=types.Content(parts=parts),
            config=types.GenerateContentConfig(
                temperature=0,  # Low temperature for precise reproduction
                max_output_tokens=2048,
                response_modalities=["TEXT","IMAGE"]  # We primarily want the image output
            )
        )

        # Process response
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    # Save and return the swapped image
                    out_path = f"/tmp/faceswap_{uuid.uuid4().hex}.png"
                    with open(out_path, "wb") as f:
                        f.write(part.inline_data.data)
                    return FileResponse(out_path)
        
        return JSONResponse(
            content={"error": "No image generated"}, 
            status_code=500
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
# @router.post("/try-on")
async def tryon_request(
    person_image: UploadFile = File(..., description="Person image"),
    overlay_cloths: List[UploadFile] = File(..., description="One or more garment images"),
    overlay_model_types: str = Form(..., description="Comma-separated model types for each garment"),
    overlay_garment_types: str = Form(..., description="Comma-separated garment categories for each garment"),
    overlay_styles: str = Form(..., description="Comma-separated styles for each garment"),
    opacity: str = Form(..., description="Enter how opaque you want the cloth to be"),
    fabric: str = Form(..., description="Enter the fabric of the cloth"),
    instructions: str = Form("", description="Any special instructions"),
):
    print(person_image.filename, type(person_image))

    # === STEP 1: Generate final image ===
    final_image = await multi_try_on(
        person_image,
        overlay_cloths,
        overlay_model_types,
        overlay_garment_types,
        overlay_styles,
        opacity,
        fabric,
        instructions,
    )

    # === STEP 2: Save image locally ===
    final_image_path = f"gemini_outputs/final_{uuid.uuid4().hex}.png"
    os.makedirs(os.path.dirname(final_image_path), exist_ok=True)

    if isinstance(final_image, Image.Image):
        final_image.save(final_image_path)

    elif isinstance(final_image, bytes):
        with open(final_image_path, "wb") as f:
            f.write(final_image)

    elif isinstance(final_image, str) and os.path.exists(final_image):
        with open(final_image, "rb") as original, open(final_image_path, "wb") as f:
            f.write(original.read())
    else:
        raise HTTPException(status_code=500, detail="Unsupported return type from multi_try_on.")

    print(f"Saved image to local: {final_image_path}")

    # === STEP 3: Upload to GCP ===
    gemini_output = generate_signed_url(
        BUCKET_NAME,
        final_image_path,
        service_account_info,
        content_type="application/octet-stream"
    )

    with open(final_image_path, "rb") as f:
        data = f.read()

    r = requests.put(gemini_output, data=data, headers={"Content-Type": "application/octet-stream"})
    if not r.ok:
        raise HTTPException(status_code=500, detail=f"GCP upload failed: {r.text}")

    person_public = f"https://storage.googleapis.com/{BUCKET_NAME}/{final_image_path}"
    print(f"Uploaded to: {person_public}")

    with open(final_image_path, "rb") as main_img:
        person_image.file.seek(0)  # rewind the original UploadFile

        files = {
            "main_image": ("final.png", main_img, "image/png"),
            "pose_image": (person_image.filename, person_image.file, person_image.content_type),
        }

        print("Sending pose request...")
        resp = requests.post(url, files=files)
        print("Received response")

    resp.raise_for_status()

    out_path = f"pose_{uuid.uuid4().hex}.png"
    with open(out_path, "wb") as fo:
        fo.write(resp.content)

    return out_path, person_public