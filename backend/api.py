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
BARS = {"scar": ["ไม่มี", "มี"], "lvef40": ["≥ 40", "< 40"], "lvef50": ["≥ 50", "< 50"]}
OUTPUT_MAP = {
    "Normal": "ไม่มีแผลเป็น",
    "Abnormal": "มีแผลเป็น",
    "geq_40": "LVEF ≥ 40",
    "leq_40": "LVEF < 40",
    "geq_50": "LVEF ≥ 50",
    "leq_50": "LVEF < 50",
}


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
            if BARS.get(model_name) is not None:
                label_lt = BARS[model_name][0]
                label_rt = BARS[model_name][1]
            else:
                label_lt, label_rt = "", ""
            prediction_output.append(
                {
                    "prediction_title": OUTPUT_MAP.get(pred["class"]),
                    "score": pred["proba"][0]
                    if model_name == "scar"
                    else pred["proba"][1],
                    "labelLt": label_lt,
                    "labelRt": label_rt,
                }
            )
        return JSONResponse(status_code=status.HTTP_200_OK, content=prediction_output)
    except Exception as e:
        print("Error in prediction:", e)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid request"},
        )
