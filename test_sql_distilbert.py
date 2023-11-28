from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch

from DistilBertModel import DistilBertModel
from wafamole.evasion import EvasionEngine

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("./sql_llm/sql_distilbert/")

inputs = tokenizer("""
    "adMiN' Or s=s#"
    """, return_tensors="pt")

with torch.no_grad():

    logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()
print(predicted_class_id)

