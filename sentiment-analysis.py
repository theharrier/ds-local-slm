from transformers import pipeline; 
import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel


def load_model():
    # Load the sentiment analysis pipeline with a specific model
    print(f"[{datetime.datetime.now()}] Loading model...")
    classifier = pipeline(task='sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')
    print(f"[{datetime.datetime.now()}] Model loaded.")
    return classifier

def get_model():
    # Lazy load the model
    if 'classifier' not in globals():
        globals()['classifier'] = load_model()
    return globals()['classifier']

def analyzeSentiment(text):
    # Get the model and perform sentiment analysis
    classifier = get_model()
    #result = classifier('hugging face is the best')
    result = classifier(text)
    print(f"[{datetime.datetime.now()}] Inference done. [{result}]")
    return result


app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/analyze-sentiment/")
async def analyze_sentiment(input_text: InputText):
    result = analyzeSentiment(input_text.text)
    #result = input_text.text + " " + str(datetime.datetime.now())
    return JSONResponse(content={"result": result})
