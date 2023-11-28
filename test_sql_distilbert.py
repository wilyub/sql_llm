import csv
import collections
from datasets import Dataset, DatasetDict
import numpy as np
import evaluate
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("./sql_llm/sql_distilbert/")
inputs = tokenizer("""
    99745017c
    """, return_tensors="pt")

with torch.no_grad():

    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
print(predicted_class_id)