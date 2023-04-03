from fastapi import FastAPI
from deepface import DeepFace
import os
import json

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": f"Welcome to ML Model API!,{os.getenv('IMG_DB_PATH')}"}

@app.get("/scan")
def scan_image():
    dfs = DeepFace.find(img_path = "images/chris_hemsworth15.png"  , db_path = "images", model_name = "Facenet512", distance_metric = 'cosine')
    # dfs -> list of dataframe
    result = json.loads(dfs[0].to_json(orient='records')) # -> list of dataframe to list of dict
    return {"data": result[0]}
