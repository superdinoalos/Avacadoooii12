from fastapi import FastAPI
import joblib
import uvicorn
from pydantic import BaseModel
avocado_app = FastAPI()

model = joblib.load('log_model_avocado (1).pkl')
scaler = joblib.load('Avocado/scaler_avocado (1).pkl')


class AvocadoSchema(BaseModel):
    firmness: float
    hue: int
    saturation: int
    brightness: int
    sound_db: int
    weight_g: int
    size_cm3: int
    color_category: str

@avocado_app.post('/predict')
async def predict(avocado: AvocadoSchema):
    avocado_dict = avocado.dict()

    new_color_category = avocado_dict.pop('color_category')
    new_color_category = [
        1 if new_color_category == 'green' else 0,
        1 if new_color_category == 'dark_green' else 0,
        1 if new_color_category == 'purple' else 0,
    ]

    features = list(avocado_dict.values()) + new_color_category
    scaled_data = scaler.transform([features])
    print(model.predict(scaled_data))
    prediction = model.predict(scaled_data)[0]
    prediction = int(prediction)
    if prediction == 1:
         prediction = 'ripe'
    elif prediction == 2:
        prediction = 'pre-conditioned'
    elif prediction == 3:
        prediction = 'hard'
    elif prediction == 4:
        prediction = 'breaking'
    elif prediction == 5:
        prediction = 'firm-ripe'

    return {'ripeness': prediction}


if __name__ == '__main__':
    uvicorn.run(avocado_app, host='127.0.0.1', port=8000)




