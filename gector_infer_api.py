from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from typing import List
from gector.gec_model import GecBERTModel
from google.cloud import storage
import os

app = FastAPI()

def download_model():
  bucket_name = "gector-api-docker-image"
  source_blob_name = "gector-model/model.th"
  destination_dir = "/tmp/models/baseline_mix_refined"
  destination_file_name = os.path.join(destination_dir, "model.th")

  os.makedirs(destination_dir, exist_ok=True)

  client = storage.Client()
  bucket = client.bucket(bucket_name)
  blob = bucket.blob(source_blob_name)

  blob.download_to_filename(destination_file_name)

  return destination_file_name

MODEL_PATH = download_model()
VOCAB_PATH = r"models/baseline_mix_refined/vocabulary"

MAX_LEN = 30
MIN_LEN = 3
ITERATION_COUNT = 4
LOWERCASE_TOKENS = 0
MODEL_NAME = "roberta"
ADDITIONAL_CONFIDENCE = 0.2
MIN_ERROR_PROBABILITY = 0.5
SPECIAL_TOKEN_FIX = 1
BATCH_SIZE = 32


model = GecBERTModel(
  vocab_path=VOCAB_PATH,
  model_paths=[MODEL_PATH],
  max_len=MAX_LEN, min_len=MIN_LEN,
  iterations=ITERATION_COUNT,
  min_error_probability=MIN_ERROR_PROBABILITY,
  lowercase_tokens=LOWERCASE_TOKENS,
  model_name=MODEL_NAME,
  special_tokens_fix=SPECIAL_TOKEN_FIX,
  log=False,
  confidence=ADDITIONAL_CONFIDENCE,
  del_confidence=0,
  is_ensemble=0,
  weigths=None
)

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
    predictions = []
    cnt_corrections = 0
    batch = []

    for sent in req.sentences:
      batch.append(sent.split())
      if len(batch) == BATCH_SIZE:
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt
        batch = []
    if batch:
      preds, cnt = model.handle_batch(batch)
      predictions.extend(preds)
      cnt_corrections += cnt
    
    result_lines = [" ".join(x) for x in predictions]
    responses = []
    for sentence in result_lines:
      nlp = spacy.load("en_core_web_sm")
      doc = nlp(sentence)
      
      is_passive = False
      for token in doc:
        if token.dep_ in ["auxpass", "nsubjpass"]:
          is_passive = True
          break

      responses.append({
        "sentence": sentence,
        "voice_type": "passive" if is_passive else "active"
      })
      
    # 1) tulis input sementara
    # with tempfile.TemporaryDirectory() as d:
    #   inp = os.path.join(d, "test_input_be.txt")
    #   outp = os.path.join(d, "test_output_best.txt")
      

    #   with open(inp, "w", encoding="utf-8") as f:
    #     for s in req.sentences:
    #       f.write(s.strip() + "\n")

    #   # 2) panggil predict.py (repo gector)
    #   cmd = [
    #     "python", "predict.py",
    #     "--model_path", MODEL_PATH,
    #     "--vocab_path", VOCAB_PATH,
    #     "--input_file", inp,
    #     "--output_file", outp,
    #     "--additional_confidence", str(ADDITIONAL_CONFIDENCE),
    #     "--min_error_probability", str(MIN_ERROR_PROBABILITY),
    #     "--special_tokens_fix", str(SPECIAL_TOKEN_FIX),
    #     "--iteration_count", str(req.iteration_count),
    #   ]
    #   p = subprocess.run(cmd, capture_output=True, text=True)

    #   if p.returncode != 0:
    #     return {"ok": False, "stderr": p.stderr, "stdout": p.stdout}

    #   # 3) baca output
    #   with open(outp, "r", encoding="utf-8") as f:
    #     preds = [line.rstrip("\n") for line in f]

    #   predictions = []
    #   for sentence in preds:
    #     nlp = spacy.load("en_core_web_sm")
    #     doc = nlp(sentence)
        
    #     is_passive = False
    #     for token in doc:
    #       is_passive = False
    #       print(token.text, token.dep_, token.pos_)
    #       if token.dep_ in ["auxpass", "nsubjpass"]:
    #         is_passive = True
    #         break

    #     predictions.append({
    #       "sentence": sentence,
    #       "voice_type": "passive" if is_passive else "active"
    #     })
    return {"ok": True, "predictions": responses}
  except Exception as e:
    return {"ok": False, "error": str(e)}
