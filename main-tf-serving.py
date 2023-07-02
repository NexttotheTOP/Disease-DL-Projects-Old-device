from fastapi import FastAPI, File, UploadFile
import logging
import uvicorn 
import numpy as np 
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()


# Configureer het logboek
logging.basicConfig(level=logging.DEBUG) 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

Model = tf.keras.models.load_model("Model_1_Potato")  
Class_names = ['Early Blight', 'Late Blight', 'Healthy']

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
    

    logger.debug("Afbeelding ontvangen: %s", file.filename)
    logger.debug("Ndarray van de afbeelding: %s", image)
    logger.debug("voorspelde klasse %s", predicted_class)

    return {"predicted_class": predicted_class}

# Verplaats de volgende regel buiten het `if __name__ == "__main__":`-blok
uvicorn_app = app

if __name__ =="__main__":
    uvicorn.run("main-tf-serving:app", host = 'localhost', port=8000, reload=True)


