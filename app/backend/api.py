from tqdm.auto import tqdm
from io import BytesIO
from PIL import Image
from typing import Optional
import numpy as np
from functools import partial
import torch

from fastapi import FastAPI, status, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import sys
import os.path as op
sys.path.append(op.join(op.dirname(__file__), "..", ".."))
from mtecg import read_bytes_to_image, remove_grid_robust, rearrange_leads
from mtecg.classifier import ECGClassifier

def preprocess_grid(image: Image.Image) -> Image.Image:
    image_no_grid = remove_grid_robust(np.array(image), n_jitter=3)
    image_rearranged = rearrange_leads(image_no_grid)
    return image_rearranged

def get_prob_for_task(output_dict: dict, task: str) -> float:
    if "lvef" in task:
        task = "lvef"
    return output_dict[task]["probability"]["positive"]

multitask_classifier = ECGClassifier(
    "models/multi-task/checkpoints/",
    model_class="multi-task",
    device="cuda" if torch.cuda.is_available() else "cpu",
    round_probabilities=True
    )

TITLE_DESC_MAP = {
    "scar": {
        "title": "Myocardial Scar",
        "description": "ความน่าจะเป็นที่จะมีแผลเป็นในกล้ามเนื้อหัวใจ",
        "average": 47.62,
    },
    "lvef50": {
        "title": "LVEF < 50",
        "description": "ความน่าจะเป็นที่ค่าประสิทธิภาพการบีบตัวของหัวใจห้องล่างซ้ายต่ำกว่า 50%",
        "average": 59.47,
    },
}


def calculate_risk_level(prob):
    """Calculate probability to risk level"""
    if prob < 30:
        return "ต่ำ"
    elif prob >= 30 and prob < 70:
        return "ปานกลาง"
    else:
        return "สูง"


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:4000",
        "http://localhost:9200",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict class from a given image bytes.

    payload: dict, dictionary with keys `image` with image bytes as value
        and `model` with model name as value (scar | lvef40 | lvef50).
    """

    if file.content_type.lower() in ["image/png", "image/jpg", "image/jpeg"]:
        request_object_content = await file.read()
        image = Image.open(BytesIO(request_object_content))
        image = preprocess_grid(image)

    elif file.content_type.lower() == "application/pdf":
        request_object_content = await file.read()
        # box=(82, 950, 3000, 2000) is the box for the new-format ECG image in the PDF.
        image = read_bytes_to_image(request_object_content, box=(82, 950, 3000, 2000))
        image = preprocess_grid(image)
    else:
        return {}

    # prediction
    try:
        # Predict only once with the multitask classifier.
        output_dict = multitask_classifier.predict(image)

        # Get output for each task.
        prediction_output = []
        for model_name in tqdm(["scar", "lvef50"]):
            title_desc = TITLE_DESC_MAP.get(model_name)
            # Get the probability and round to 2 decimal places.
            probability = round(get_prob_for_task(output_dict, task=model_name) * 100, 2)
            prediction_output.append(
                {
                    "title": title_desc["title"],
                    "description": title_desc["description"],
                    "average": title_desc["average"],
                    "probability": probability,
                    "risk_level": calculate_risk_level(probability),
                }
            )
        return JSONResponse(status_code=status.HTTP_200_OK, content=prediction_output)
    except Exception as e:
        print("Error in prediction:", e)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid request"},
        )
