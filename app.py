from transformers import (
    AutoModelForSequenceClassification,
    DistilBertTokenizer,
    TextClassificationPipeline,
    pipeline
)
from summarizer import Summarizer
import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()
class Input(BaseModel):
    input:List[Dict]

@app.get("/")
def hello():
    return {"result":"Hello World"}

@app.post('/compute')
def compute_highlights(input:Input):
    print(input)
    boundary = topic_boundaries(input.input)
    return get_title(boundary,input.input)

def topic_boundaries(input):
    model_name = "BlueOrangeDigital/distilbert-cross-segment-document-chunking"
    model_path = "/model"
    id2label = {0: "SAME", 1: "DIFFERENT"}
    label2id = {"SAME": 0, "DIFFERENT": 1}

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name,cache_dir=model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        cache_dir=model_path
    )
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
    boundary = []
    for i in range(len(input)-1):
        pairs = [f"{input[i]['text']} [SEP] {input[i+1]['text']}"]
        result = pipe(pairs)
        if result[0][0]['score']<result[0][1]['score']:
            boundary.append(i)
    new_boundary = []
    for i in range(len(boundary)-1):
        if (boundary[i+1]-boundary[i])<=10:
            continue
        new_boundary.append(boundary[i])
    print("BOUNDARY:--  ",new_boundary)
    return new_boundary

def get_title(boundary,input):
    model_name = "czearing/article-title-generator"
    pipe = pipeline("text2text-generation", model=model_name)
    model = Summarizer()
    
    output = []
    for i in range(len(boundary)-1):
        text = ""
        start_time = input[boundary[i]]["offset"]
        end_time = start_time
        for j in range(boundary[i],boundary[i+1]):
            text += input[j]["text"]
            end_time+=input[j]["duration"]
        if len(text)>512:
            text =  model(text, min_length=60)
        title = pipe(text)
        output.append({"start":start_time,"end":end_time,"text":title[0]["generated_text"]})
    print("OUTPUT:--  ",output)
    return output


