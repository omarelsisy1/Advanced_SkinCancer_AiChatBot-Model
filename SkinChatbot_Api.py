import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle

# Load the saved model
model = tf.keras.models.load_model("/Users/Downloads/Skin_cancer_chatbot_model.h5")

# Load the tokenizer directly from JSON
with open("/Users/Downloads/chatbot/tokenizer.json") as f:
    tokenizer_data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tokenizer_data))

# Load the label encoder
with open("/Users/Downloads/chatbot/label_encoder.pkl", 'rb') as f:
    label_encoder = pickle.load(f)

# Load responses
with open("/Users/Downloads/chatbot/intentsNEW2.json") as file:
    data = json.load(file)
    responses = {intent['tag']: intent['responses'] for intent in data['intents']}

# Define a FastAPI app
app = FastAPI()

# Define a request body model
class Message(BaseModel):
    text: str

# Function to get response
def get_response(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=13, padding='post')  # Adjust maxlen to the expected length
    predicted_tag = model.predict(padded_sequence)
    tag = label_encoder.inverse_transform([np.argmax(predicted_tag)])
    return np.random.choice(responses[tag[0]])

@app.post("/get_response/")
async def respond(message: Message):
    try:
        response = get_response(message.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
