from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
import traceback
import os
import uuid
import json
from typing import List
from PIL import Image
import io
from google import genai
from google.genai import types


load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

async def multi_try_on(
    person_image: UploadFile = File(..., description="Person image"),
    overlay_cloths: List[UploadFile] = File(..., description="One or more garment images"),
    overlay_model_types: str = Form(..., description="Comma-separated model types for each garment"),
    overlay_garment_types: str = Form(..., description="Comma-separated garment categories for each garment"),
    overlay_styles: str = Form(..., description="Comma-separated styles for each garment"),
    opacity: str = Form(..., description="Enter how opaque you want the cloth to be"),
    fabric: str = Form(..., description="Enter the fabric of the cloth"),
    instructions: str = Form("", description="Any special instructions")
):
    try:
        if instructions == "":
          instructions = "Make sure that the cloths are appropriately overlay on the person and the face should be exactly the same as the original."

        print(f"{instructions=}")
        N = len(overlay_cloths)
        model_types = [s.strip() for s in overlay_model_types.split(",")]
        garment_types = [s.strip() for s in overlay_garment_types.split(",")]
        styles = [s.strip() for s in overlay_styles.split(",")]

        if not (len(model_types) == len(garment_types) == len(styles) == N):
            raise HTTPException(
                status_code=400,
                detail="overlay_model_types, overlay_garment_types, overlay_styles must each have exactly as many comma-separated items as overlay_cloths"
            )

        person_bytes = await person_image.read()
        cloth_bytes_list = []
        for f in overlay_cloths:
            b = await f.read()
            cloth_bytes_list.append(b)            

        base = {
          "instructions": """
            Generate a photorealistic virtual try-on where the clothing items perfectly match the body structure and proportions of the input person. 
            Ensure:

            1. FACE PRESERVATION (ABSOLUTE PRIORITY):
            - Copy and paste the face from input with absolutely no modification
            - Match skin texture, expression, and features pixel-for-pixel
            - Allow only minimal blending at neck/garment boundaries
            - Do not synthesize, redraw, or reinterpret any facial features
            - Preserve all micro-details: pores, wrinkles, facial hair exactly as original
            
            1. Body Structure Preservation:
            - Maintain the exact body proportions, posture, and silhouette from the input image
            - Precisely follow the body landmarks and pose map without any deviation
            - Keep all limb positions, angles, and body contours identical to the source

            2. Garment Fitting:
            - Overlay each garment to match the body's exact dimensions and shape
            - Maintain original clothing proportions - don't stretch or distort patterns/logos
            - Preserve all garment details (stitching, textures, patterns) exactly as provided
            - Ensure realistic fabric draping that conforms to the body's posture

            3. Natural Integration:
            - Blend garment edges seamlessly with the body while maintaining sharp details
            - Match lighting and shadows to the original image's environment
            - Maintain realistic fabric behavior (wrinkles where expected, tension points)

            Important Constraints:
            - DO NOT change the face of the input person. It must be PIXEL LOCKED
            - DO NOT Mix the faces of the people.
            - DO NOT alter body proportions or posture in any way
            - DO NOT modify the intrinsic properties of the garments
            - DO NOT allow clothing to appear "pasted on" - maintain depth and realism
            - DO NOT change the person's scale or positioning in frame
          """
        }

        for i in range(N):
            idx = i + 2
            base[f"garment_image_{idx}"] = {
                "description": f"Clothing item #{i+1}: used for color, pattern, texture, style",
                "id": f"input_{idx}"
            }

        metadata_lines = []
        for i in range(N):
            metadata_lines.append(
                f"- Garment {i+1}: ModelType={model_types[i]}, Category={garment_types[i]}, Style={styles[i]}"
            )

        for i in range(N):
            metadata_lines.append(
                f"The fabric of cloth {i} is {fabric[i]} and it must be {opacity[i]}% opaque. This must be strictly followed."
            )

        prompt_text = f"""
            {json.dumps(base, indent=2)}

            You are a virtual fashion stylist.
            Create a realistic try‑on visualization of the uploaded clothing(s) onto the person image.
            Metadata:
            - Person ID: input_1
            {"".join(metadata_lines)}

            Special Instructions: {instructions}
        """

        parts = [
          prompt_text,
          types.Part(inline_data=types.Blob(mime_type=person_image.content_type, data=person_bytes))
        ]

        for i, cloth in enumerate(overlay_cloths):
            parts.append(
              types.Part(inline_data=types.Blob(
                mime_type=cloth.content_type,
                data=cloth_bytes_list[i]
              ))
            )

        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=parts,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,
                response_modalities=["TEXT", "IMAGE"]
            )
        )

        print(response)

        caption = "No description."
        image_bytes = None
        if response.candidates:
            for p in response.candidates[0].content.parts:
                if p.text:
                    caption = p.text.strip()
                elif p.inline_data:
                    image_bytes = p.inline_data.data

        result = {"text": caption}
        if image_bytes:
            out_path = f"/tmp/tryon_{uuid.uuid4().hex}.png"
            with open(out_path, "wb") as fo:
                fo.write(image_bytes)
            return out_path
        
        return JSONResponse(content=result)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
