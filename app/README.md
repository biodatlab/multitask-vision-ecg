# Application

We implement an application to predict myocardial scar (MS) and LVEF.
Our application contains `frontend` which stores frontend application and `backend` which stores
APIs implemented using [FastAPI](https://fastapi.tiangolo.com/).

## Frontend

The frontend application is located in the `frontend` folder. For convenience, install
the dependencies for the frontend using a single command. Note that we require Node version `>= 14`.

```sh
cd frontend
npm run setup
```

You can run this command in the root of the project to serve both applications at once:

```sh
npm run demo # run the frontend and backend together
```

Or you can run the frontend separately:

```sh
npm run dev
```

## Backend

Install dependencies using `pip` and `conda`.

```sh
cd backend
pip install -r requirements.txt

# for mac user, poppler need to be installed via conda
conda install -c conda-forge poppler
```

Then, serve the backend using `uvicorn`.

```sh
uvicorn api:app --reload
```
