import os
import uuid
import json
import base64
import time
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import Optional, Literal, List
import fal_client
import requests
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import euclidean

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
from google.auth.crypt import RSASigner
from dotenv import load_dotenv
from routers.auth import get_current_user

load_dotenv()

FAL_AI_API_URL = "https://fal.ai/models/fal-ai/bria/background/replace/playground"
API_KEY = os.getenv("FAL_KEY")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URI)
db = client["fashion_app"]

body_size_coll       = db["body_size_results"]
recommend_coll       = db["size_recommendations"]
bg_prompt_coll       = db["bg_replace_prompt_jobs"]
bg_image_coll        = db["bg_replace_image_jobs"]

with open(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")) as f:
    service_account_info = json.load(f)

BUCKET_NAME = "ai-fashion-app-mirror-me"

router = APIRouter(prefix="/body-size", tags=["Body Size Detection"], dependencies=[Depends(get_current_user)])

class BackgroundReplaceResponse(BaseModel):
    success: bool
    result_url: Optional[str] = None
    error: Optional[str] = None
    request_id: Optional[str] = None

class BodyInfo(BaseModel):
    gender: str
    height: float
    weight: float 
    shoulder: Optional[float] = None  
    chest: Optional[float] = None  
    hip: Optional[float] = None 

def generate_signed_url(bucket_name, object_name, svc_info, expiration=15*60, content_type="application/octet-stream"):
    method = "PUT"
    expires_at = int(time.time() + expiration)
    uri = f"/{bucket_name}/{object_name}"
    to_sign = "\n".join([method, "", content_type, str(expires_at), uri])
    signer = RSASigner.from_string(svc_info["private_key"])
    signature = base64.b64encode(signer.sign(to_sign.encode())).decode()
    return (
        f"https://storage.googleapis.com{uri}"
        f"?GoogleAccessId={svc_info['client_email']}"
        f"&Expires={expires_at}"
        f"&Signature={requests.utils.quote(signature)}"
    )


class BodySizeDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, 
                                     model_complexity=2, 
                                     min_detection_confidence=0.5)
        
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        self.size_thresholds = {
            'shoulder_hip_ratio': {
                'XS': (0.85, 0.95),
                'S': (0.95, 1.05),
                'M': (1.05, 1.15),
                'L': (1.15, 1.25),
                'XL': (1.25, 1.40)
            },
            'chest_hip_ratio': {
                'XS': (0.80, 0.90),
                'S': (0.90, 1.00),
                'M': (1.00, 1.10),
                'L': (1.10, 1.20),
                'XL': (1.20, 1.35)
            }
        }

    def extract_pose_landmarks(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks)
        return None
    
    def get_person_segmentation(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.selfie_segmentation.process(rgb_image)
        
        if results.segmentation_mask is not None:
            mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.medianBlur(mask, 5)
            return mask
        
        return self.get_improved_color_segmentation(image)
    
    def get_improved_color_segmentation(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([25, 255, 255])
        mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
        
        lower_clothing = np.array([0, 0, 20])
        upper_clothing = np.array([180, 255, 200])
        mask_clothing = cv2.inRange(hsv, lower_clothing, upper_clothing)
        
        lower_lab = np.array([20, 20, 20])
        upper_lab = np.array([255, 255, 255])
        mask_lab = cv2.inRange(lab, lower_lab, upper_lab)
        
        combined_mask = cv2.bitwise_or(mask_skin, mask_clothing)
        combined_mask = cv2.bitwise_or(combined_mask, mask_lab)
        
        mask_grabcut = self.apply_grabcut_refinement(image, combined_mask)
        return mask_grabcut
    
    def apply_grabcut_refinement(self, image, initial_mask):
        mask = np.zeros(initial_mask.shape[:2], np.uint8)
        mask[initial_mask == 255] = 3
        mask[initial_mask == 0] = 0
        
        height, width = image.shape[:2]
        rect = (int(width*0.1), int(height*0.1), int(width*0.8), int(height*0.8))
        
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            final_mask = final_mask * 255
            kernel = np.ones((3, 3), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
            return final_mask
        except:
            kernel = np.ones((5, 5), np.uint8)
            cleaned_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
            return cleaned_mask
    
    def find_line_mask_intersections(self, mask, line_y, image_width):
        if line_y < 0 or line_y >= mask.shape[0]:
            return None, None
        
        row_pixels = mask[line_y, :]
        segments = []
        start = None
        
        for i, pixel in enumerate(row_pixels):
            if pixel > 0 and start is None:
                start = i
            elif pixel == 0 and start is not None:
                segments.append((start, i-1))
                start = None
        
        if start is not None:
            segments.append((start, len(row_pixels)-1))
        
        if segments:
            largest_segment = max(segments, key=lambda x: x[1] - x[0])
            return float(largest_segment[0]), float(largest_segment[1])
        
        return None, None
    
    def calculate_body_measurements(self, image):
        height, width = image.shape[:2]
        landmarks = self.extract_pose_landmarks(image)
        if landmarks is None:
            raise ValueError("Could not detect pose landmarks")
        
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        left_shoulder_px = [int(left_shoulder[0] * width), int(left_shoulder[1] * height)]
        right_shoulder_px = [int(right_shoulder[0] * width), int(right_shoulder[1] * height)]
        left_hip_px = [int(left_hip[0] * width), int(left_hip[1] * height)]
        right_hip_px = [int(right_hip[0] * width), int(right_hip[1] * height)]
        
        shoulder_distance = euclidean(left_shoulder_px, right_shoulder_px)
        hip_level = (left_hip_px[1] + right_hip_px[1]) // 2
        chest_level = (left_shoulder_px[1] + right_shoulder_px[1]) // 2 + 30
        
        mask = self.get_person_segmentation(image)
        if mask is None or np.sum(mask) == 0:
            raise ValueError("Could not detect person in the image")
        
        hip_left, hip_right = self.find_line_mask_intersections(mask, hip_level, width)
        chest_left, chest_right = self.find_line_mask_intersections(mask, chest_level, width)
        
        hip_distance = abs(hip_right - hip_left) if hip_left is not None and hip_right is not None else 0
        chest_distance = abs(chest_right - chest_left) if chest_left is not None and chest_right is not None else 0
        
        return {
            'measurements': {
                'shoulder_distance': shoulder_distance,
                'hip_distance': hip_distance,
                'chest_distance': chest_distance
            }
        }
    
    def classify_size(self, measurements):
        shoulder_dist = measurements['shoulder_distance']
        hip_dist = measurements['hip_distance']
        chest_dist = measurements['chest_distance']
        
        if hip_dist == 0 or chest_dist == 0:
            return "Unknown"
        
        shoulder_hip_ratio = shoulder_dist / hip_dist
        chest_hip_ratio = chest_dist / hip_dist
        
        size_scores = {}
        
        for size in ['XS', 'S', 'M', 'L', 'XL']:
            score = 0
         
            shr_min, shr_max = self.size_thresholds['shoulder_hip_ratio'][size]
            if shr_min <= shoulder_hip_ratio <= shr_max:
                score += 1
            else:
                if shoulder_hip_ratio < shr_min:
                    score -= (shr_min - shoulder_hip_ratio) * 2
                else:
                    score -= (shoulder_hip_ratio - shr_max) * 2

            chr_min, chr_max = self.size_thresholds['chest_hip_ratio'][size]
            if chr_min <= chest_hip_ratio <= chr_max:
                score += 1
            else:
                if chest_hip_ratio < chr_min:
                    score -= (chr_min - chest_hip_ratio) * 2
                else:
                    score -= (chest_hip_ratio - chr_max) * 2
            
            size_scores[size] = score
        
        return max(size_scores, key=size_scores.get)

detector = BodySizeDetector()

@router.post("/detect", summary="Detect body size")
async def detect_body_size(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    uid = str(current_user["_id"])
    job_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename)
    gcs_key = f"body_size/input/{job_id}{ext}"
    signed = generate_signed_url(BUCKET_NAME, gcs_key, service_account_info, content_type=file.content_type)
    data = await file.read()
    resp = requests.put(signed, data=data, headers={"Content-Type": file.content_type})
    if not resp.ok:
        raise HTTPException(status_code=500, detail="Failed to upload input image")
    input_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{gcs_key}"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    img = cv2.imread(tmp_path)
    os.unlink(tmp_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    measurements = detector.calculate_body_measurements(img)["measurements"]
    size = detector.classify_size(measurements)

    await body_size_coll.insert_one({
        "job_id": job_id,
        "user_id": uid,
        "input_image_url": input_url,
        "measurements": measurements,
        "size": size,
        "created_at": datetime.now(timezone.utc)
    })

    return {"size": size, "input_url": input_url}

@router.post("/recommend", summary="Recommend size from gender, height, weight and measurements")
async def recommend_size(
    body: BodyInfo,
    current_user: dict = Depends(get_current_user),
):
    uid = str(current_user["_id"])
    job_id = str(uuid.uuid4())

    try:
        bmi = body.weight / ((body.height / 100) ** 2)

        def size_by_bmi(bmi_val: float, gender: str) -> str:
            if gender.lower() == "male":
                if bmi_val < 18.5: return "XS"
                elif bmi_val < 21.5: return "S"
                elif bmi_val < 24.5: return "M"
                elif bmi_val < 27.5: return "L"
                else: return "XL"
            else: 
                if bmi_val < 17.5: return "XS"
                elif bmi_val < 20.5: return "S"
                elif bmi_val < 23.5: return "M"
                elif bmi_val < 26.5: return "L"
                else: return "XL"

        bmi_size = size_by_bmi(bmi, body.gender)

        if all([body.shoulder, body.chest, body.hip]):
            measurements = {
                "shoulder_distance": body.shoulder,
                "chest_distance": body.chest,
                "hip_distance": body.hip,
            }
            refined_size = detector.classify_size(measurements)
            result_size = refined_size
            method = "refined by measurements"
        else:
            result_size = bmi_size
            method = "estimated by BMI"

        await recommend_coll.insert_one({
            "job_id": job_id,
            "user_id": uid,
            "input": body.dict(),
            "size": result_size,
            "method": method,
            "created_at": datetime.now(timezone.utc)
        })

        return {"size": result_size, "method": method}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/replace-background", response_model=BackgroundReplaceResponse, summary="Replace background using prompt and/or reference image")
async def replace_background(
    original_image: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(""),
    background_image: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user),
):
    if not prompt and not background_image:
        raise HTTPException(400, detail="Provide at least a prompt or background image")

    uid = str(current_user["_id"])
    job_id = str(uuid.uuid4())

    _, ext_o = os.path.splitext(original_image.filename)
    key_o = f"bg/input_original/{job_id}{ext_o}"
    signed_o = generate_signed_url(BUCKET_NAME, key_o, service_account_info, content_type=original_image.content_type)
    data_o = await original_image.read()
    r1 = requests.put(signed_o, data=data_o, headers={"Content-Type": original_image.content_type})
    if not r1.ok:
        raise HTTPException(500, "Failed to upload original image")
    url_o = f"https://storage.googleapis.com/{BUCKET_NAME}/{key_o}"

    url_b = None
    if background_image:
        _, ext_b = os.path.splitext(background_image.filename)
        key_b = f"bg/input_background/{job_id}{ext_b}"
        signed_b = generate_signed_url(BUCKET_NAME, key_b, service_account_info, content_type=background_image.content_type)
        data_b = await background_image.read()
        r2 = requests.put(signed_b, data=data_b, headers={"Content-Type": background_image.content_type})
        if not r2.ok:
            raise HTTPException(500, "Failed to upload background image")
        url_b = f"https://storage.googleapis.com/{BUCKET_NAME}/{key_b}"

    fal_args = {
        "image_url": url_o,
        "num_images": 1
    }
    if prompt:
        fal_args["prompt"] = prompt
        fal_args["negative_prompt"] = negative_prompt
    if url_b:
        fal_args["ref_image_url"] = url_b
        if not prompt:
            fal_args["prompt"] = "Replace background with reference"

    handler = await fal_client.submit_async("fal-ai/bria/background/replace", arguments=fal_args)
    result = await handler.get()

    fal_cdn_url = result.get("images", [{}])[0].get("url") or result.get("image")
    if not fal_cdn_url:
        raise HTTPException(500, "Fal AI did not return an output image")

    img_resp = requests.get(fal_cdn_url)
    if not img_resp.ok:
        raise HTTPException(500, "Failed to fetch result image from Fal")

    key_out = f"bg/output/{job_id}.png"
    signed_out = generate_signed_url(BUCKET_NAME, key_out, service_account_info, content_type="image/png")
    r3 = requests.put(signed_out, data=img_resp.content, headers={"Content-Type": "image/png"})
    if not r3.ok:
        raise HTTPException(500, "Failed to upload result image")

    own_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{key_out}"

    record = {
        "job_id": job_id,
        "user_id": uid,
        "input_url": url_o,
        "output_url": own_url,
        "request_id": handler.request_id,
        "created_at": datetime.now(timezone.utc),
    }
    if prompt:
        record["prompt"] = prompt
        record["negative_prompt"] = negative_prompt
    if url_b:
        record["background_url"] = url_b

    await bg_prompt_coll.insert_one(record)

    return BackgroundReplaceResponse(success=True, result_url=own_url, request_id=handler.request_id)
