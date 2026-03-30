import io
from PIL import Image
from weather import get_weather
from prediction_script import predict_paddy
from fastapi import FastAPI, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict_paddy")
async def predict_paddy_state(file: UploadFile = File(...)):
    request_object_content = await file.read()
    img                    = Image.open(io.BytesIO(request_object_content))
    result                 = predict_paddy(img)
    return jsonable_encoder(result)

@app.post("/city_weather")
async def get_city_weather(city: str):
    return jsonable_encoder(get_weather(city))

