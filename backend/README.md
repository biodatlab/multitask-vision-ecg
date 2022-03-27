# Backend

This folder contains small a library for processing ECG files
and FastAPI for model prediction.

## ECG utilities

```py
from ecg_utils import (
    read_pdf_to_image,
    load_learner_path
)
learner = load_learner_path("model.pkl")
prediction = predict_ecg(learner, path)
```

##
