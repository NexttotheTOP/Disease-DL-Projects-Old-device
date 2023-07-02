from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn 
import numpy as np 
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# model_path = "C:/Users/woutv/Documents/Potato-Disease/Training_Potato/Models/Model_1_Potato"
# model = tf.keras.models.load_model(model_path)


Model = tf.keras.models.load_model("Model_1_Potato")   
#C:\Users\woutv\Documents\Potato-Disease\Training_Potato\Models\Model_1_Potato
Class_names = ['Early Blight', 'Late Blight', 'Healthy']

# Configureer het logboek
logging.basicConfig(level=logging.DEBUG) 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@app.get("/ping")
async def ping():
    return "Hello baby, I'am aliveee"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.debug('Ontvangen foto: %s', file.filename)
    #file: UploadFile = File(...)

    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0) # (256, 256, 3)
    
    prediction = Model.predict(img_batch)
    predicted_class_index = np.argmax(prediction)
    predicted_class = Class_names[predicted_class_index]
    confidence_score = float(prediction[0][predicted_class_index])

    

    logger.debug("Afbeelding ontvangen: %s", file.filename)
    logger.debug("Ndarray van de afbeelding: %s", image)
    logger.debug("voorspelde klasse %s", predicted_class)
    logger.debug("confidence Score %s", confidence_score)

    return {"predicted_class": predicted_class, "confidence_score": confidence_score}

# Verplaats de volgende regel buiten het `if __name__ == "__main__":`-blok
# uvicorn_app = app

if __name__ =="__main__":
    uvicorn.run("main:app", host = 'localhost', port=8000, reload=True)


