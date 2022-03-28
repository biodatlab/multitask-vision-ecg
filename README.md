# Vision-based ECG Classification using Convolutional Neural Network

Classifying 12 lead Electrocardiogram (EKG) using Convolutional Neural Network (CNN).
We train deep neural network (CNN) to classify image for 12-leads ECG scan.
Data are provided from Siriraj Hospital, Thailand.

## Running web application

The repository contains `frontend` and `backend` folders. Run both folders separately to
serve an application.

### Frontend

```sh
npm install
```

### Backend

Install Python dependencies and run FastAPI as follows

```sh
pip install -r requirements.txt
uvicorn api:app --reload
```
