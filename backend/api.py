from tqdm.auto import tqdm
from io import BytesIO
from PIL import Image
from ecg_utils import *
from typing import Optional

from fastapi import FastAPI, status, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


MODEL_DICT = {
    "scar": "models/scar_model.pkl",
    "lvef40": "models/lvef40_model.pkl",
    "lvef50": "models/lvef50_model.pkl",
}
MODELS = {k: load_learner_path(v) for k, v in MODEL_DICT.items()}
TITLE_DESC_MAP = {
    "scar": {
        "title": "Myocardial Scar",
        "description": "ความน่าจะเป็นที่จะมีแผลเป็นที่กล้ามเนื้อหัวใจ",
        "average": 47.62
    },
    "lvef40": {
        "title": "LVEF < 40",
        "description": "ความน่าจะเป็นที่ค่าประสิทธิภาพการทำงานของหัวใจห้องล่างซ้ายต่ำกว่า 40 %",
        "average": 59.47
    },
    "lvef50": {
        "title": "LVEF < 50",
        "description": "ความน่าจะเป็นที่ค่าประสิทธิภาพการทำงานของหัวใจห้องล่างซ้ายต่ำกว่า 50 %",
        "average": 59.47
    }
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
        image = np.array(image)[:, :, :3]
    elif file.content_type.lower() == "application/pdf":
        request_object_content = await file.read()
        image = read_bytes_to_image(request_object_content)
        image = np.array(image)
    else:
        return {}

    # prediction
    try:
        prediction_output = []
        for model_name in tqdm(["scar", "lvef40", "lvef50"]):
            pred = predict_array(MODELS[model_name], image)
            title_desc = TITLE_DESC_MAP.get(model_name)
            probability = pred["proba"][0] * 100
            prediction_output.append({
                "title": title_desc["title"],
                "description": title_desc["description"],
                "average": title_desc["average"],
                "probability": probability,
                "risk_level": calculate_risk_level(probability)
            })
        return JSONResponse(status_code=status.HTTP_200_OK, content=prediction_output)
    except Exception as e:
        print("Error in prediction:", e)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid request"},
        )
