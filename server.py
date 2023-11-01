from fastapi import FastAPI
import torch
from model import GRURNN
from tokenizer import Tokenizer

app = FastAPI()

model = GRURNN(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('path/to/weights.pth'))
model.eval()

tokenizer = Tokenizer("path/to/spm.model")

@app.get("/predict/")
def predict(text: str):
    tokenized_text = tokenizer.encode(text)
    # Perform prediction with the model
    # Return the result
