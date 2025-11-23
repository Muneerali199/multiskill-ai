import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

FINETUNED_DIR = "multiskill_distilgpt2_model"

app = FastAPI(title="Multi-Skill distilgpt2 Finetuned API")

class GenerateRequest(BaseModel):
    domain: str
    instruction: str
    max_new_tokens: int = 160

print("Loading finetuned model from", FINETUNED_DIR)
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)
model = AutoModelForCausalLM.from_pretrained(FINETUNED_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print("Model loaded on device:", device)

@app.get("/")
def root():
    return {"status": "ok", "message": "Multi-skill distilgpt2 finetuned API is running."}

@app.post("/generate")
def generate(req: GenerateRequest):
    prompt = (
        "[INST] You are an expert assistant in the domain of " + req.domain +
        ". Follow the instruction carefully.\n\nInstruction: " +
        req.instruction + " [/INST]\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}
