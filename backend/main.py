from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Allow CORS for all origins (or limit it to your frontend host)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and dataframe
pipe = pickle.load(open("pipe.pkl", "rb"))
df = pickle.load(open("df.pkl", "rb"))

class LaptopFeatures(BaseModel):
    company: str
    types: str
    ram: int
    weight: float
    touchscreen: str
    ips: str
    screen_size: float
    resolution: str
    cpu: str
    hdd: int
    ssd: int
    gpu: str
    os: str

@app.post("/predict")
def predict(data: LaptopFeatures):
    touchscreen = 1 if data.touchscreen == 'Yes' else 0
    ips = 1 if data.ips == 'Yes' else 0
    X_res, Y_res = map(int, data.resolution.split('x'))
    ppi = ((X_res**2 + Y_res**2) ** 0.5) / data.screen_size

    query = np.array([
        data.company,
        data.types,
        data.ram,
        data.weight,
        touchscreen,
        ips,
        ppi,
        data.cpu,
        data.hdd,
        data.ssd,
        data.gpu,
        data.os
    ]).reshape(1, 12)

    pred = int(np.exp(pipe.predict(query)[0]))
    return {"predicted_price": pred}

@app.get("/options")
def get_options():
    return {
        "company": sorted(df["Company"].unique().tolist()),
        "types": sorted(df["TypeName"].unique().tolist()),
        "cpu": sorted(df["Cpu brand"].unique().tolist()),
        "gpu": sorted(df["Gpu brand"].unique().tolist()),
        "os": sorted(df["os"].unique().tolist()),
        "resolution": [
            "1920x1080", "1366x768", "1600x900", "3840x2160",
            "3200x1800", "2880x1800", "2560x1600", "2560x1440", "2304x1440"
        ],
        "ram": [2, 4, 6, 8, 12, 16, 24, 32, 64],
        "hdd": [0, 128, 256, 512, 1024, 2048],
        "ssd": [0, 8, 128, 256, 512, 1024]
    }
