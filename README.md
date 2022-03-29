# Vision-based ECG Classification using Convolutional Neural Network

Classifying 12 lead Electrocardiogram (EKG) using Convolutional Neural Network (CNN).
We train deep neural network (CNN) to classify image for 12-leads ECG scan.
Data are provided from Siriraj Hospital, Thailand.

## Requirements

- Python (preferred Conda) >= 3.7
- NodeJS >= 14.x.x

For convenience, installing runner for the project and dependencies for the frontend is included in a single command:

```sh
npm run setup
```

But you will still need to install the dependencies for the backend separately:

```sh
cd backend
pip install -r requirements.txt

# for mac user, poppler need to be installed via conda
conda install -c conda-forge poppler
```

## Running the project

The repository contains `frontend` and `backend` applications. You can run this command in the root of the project to serve both applications at once:

```sh
npm run demo
```

You can run each application separately if you want:

### Frontend

```sh
cd frontend
npm run dev
```

### Backend

```sh
cd backend
uvicorn api:app --reload
```
