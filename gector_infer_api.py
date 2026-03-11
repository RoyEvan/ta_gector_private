from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import spacy
import tempfile
import os
from typing import List

app = FastAPI()

MODEL_PATH = r"models/finetuned/finetuned/model.th"
VOCAB_PATH = r"data/output_vocabulary"
ADDITIONAL_CONFIDENCE = 0.2
MIN_ERROR_PROBABILITY = 0.5
SPECIAL_TOKEN_FIX = 1

class Req(BaseModel):
  sentences: List[str]
  iteration_count: int = 3
  
def convert_passive_to_active(doc):
  """Attempt to convert a passive sentence to active voice."""
  subject = None
  agent = None
  verb = None
  aux = None
  
  for token in doc:
    print(token.text, token.dep_, token.pos_)
    if token.dep_ == "nsubjpass":
      subject = token
    elif token.dep_ == "agent":
      for child in token.children:
        if child.dep_ == "pobj":
          agent = child
    elif token.dep_ == "auxpass":
      aux = token
    elif token.pos_ == "VERB" and token.dep_ == "ROOT":
      verb = token
  
  if subject and verb:
    # Get base form of verb
    verb_base = verb.lemma_
    agent_text = agent.text if agent else "someone"
    
    # Simple conversion: Agent + verb (base) + subject
    return f"{agent_text.capitalize()} {verb_base}s {subject.text}."
  
  return None

def convert_active_to_passive(doc):
  """Attempt to convert an active sentence to passive voice."""
  subject = None
  obj = None
  verb = None
  
  for token in doc:
    if token.dep_ == "nsubj":
      subject = token
    elif token.dep_ in ["dobj", "obj"]:
      obj = token
    elif token.pos_ == "VERB" and token.dep_ == "ROOT":
      verb = token
  
  if subject and obj and verb:
    # Get past participle form (not always correct)
    verb_pp = verb.lemma_ + "ed"  
    
    # Simple conversion: Object + was/were + past participle + by + subject
    be_verb = "was" if obj.tag_ in ["NN", "NNP"] else "were"
    return f"{obj.text.capitalize()} {be_verb} {verb_pp} by {subject.text}."
  
  return None

@app.post("/infer")
def infer(req: Req):
  try:
    # 1) tulis input sementara
    with tempfile.TemporaryDirectory() as d:
      inp = os.path.join(d, "test_input_be.txt")
      outp = os.path.join(d, "test_output_best.txt")

      with open(inp, "w", encoding="utf-8") as f:
        for s in req.sentences:
          f.write(s.strip() + "\n")

      # 2) panggil predict.py (repo gector)
      cmd = [
        "python", "predict.py",
        "--model_path", MODEL_PATH,
        "--vocab_path", VOCAB_PATH,
        "--input_file", inp,
        "--output_file", outp,
        "--additional_confidence", str(ADDITIONAL_CONFIDENCE),
        "--min_error_probability", str(MIN_ERROR_PROBABILITY),
        "--special_tokens_fix", str(SPECIAL_TOKEN_FIX),
        "--iteration_count", str(req.iteration_count),
      ]
      p = subprocess.run(cmd, capture_output=True, text=True)

      if p.returncode != 0:
        return {"ok": False, "stderr": p.stderr, "stdout": p.stdout}

      # 3) baca output
      with open(outp, "r", encoding="utf-8") as f:
        preds = [line.rstrip("\n") for line in f]

      predictions = []
      for sentence in preds:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentence)
        
        is_passive = False
        for token in doc:
          is_passive = False
          print(token.text, token.dep_, token.pos_)
          if token.dep_ in ["auxpass", "nsubjpass"]:
            is_passive = True
            break

        predictions.append({
          "sentence": sentence,
          "voice_type": "passive" if is_passive else "active"
        })
    return {"ok": True, "predictions": predictions}
  except Exception as e:
    return {"ok": False, "error": str(e)}
