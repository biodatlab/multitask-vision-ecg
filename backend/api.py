from ecg_utils import *
from typing import Optional

from fastapi import FastAPI, status, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


MODEL_DICT = {
    "scar": "models/scar_model.pkl",
    "lvef40": "models/lvef50_model.pkl",
    "lvef50": "models/lvef40_model.pkl",
}
MODELS = {k: load_learner_path(v) for k, v in MODEL_DICT.items()}


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
    return {"filename": file.filename, "file": file}
    # model_name = payload.get("model", None)
    # if payload is not None and MODELS.get(model_name) is not None:
    #     image = read_bytes_to_image(payload["image"])
    #     image = np.array(image)
    #     pred = predict_array(MODELS[model_name], image)
    #     return JSONResponse(status_code=status.HTTP_200_OK, content=pred)
    # else:
    #     return JSONResponse(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         content={"error": "Invalid request"},
    #     )
