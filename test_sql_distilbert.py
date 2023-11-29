from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from datasets import load_from_disk

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model_sql = DistilBertForSequenceClassification.from_pretrained("./sql_llm/sql_distilbert/")
sql_dataset = load_from_disk("./sql_ds.hf/")
test_set = sql_dataset['test']
ground_truth = np.zeros(test_set.num_rows)
predicted_labels_model = np.zeros(test_set.num_rows)
predicted_labels_sql_model = np.zeros(test_set.num_rows)
for i, entry in enumerate(tqdm(test_set, total=test_set.num_rows)):
    inputs = tokenizer(entry['Query'], return_tensors="pt")
    ground_truth[i] = entry['label']
    with torch.no_grad():
        logits = model(**inputs).logits
        sql_logits = model_sql(**inputs).logits
    predicted_class_id = logits.argmax().item()
    predicted_labels_model[i] = predicted_class_id
    predicted_class_id_sql = sql_logits.argmax().item()
    predicted_labels_sql_model[i] = predicted_class_id_sql

print("Distilbert Accuracy: " + str(accuracy_score(ground_truth, predicted_labels_model)))
print("SQL Distilbert Accuracy: " + str(accuracy_score(ground_truth, predicted_labels_sql_model)))

# inputs = tokenizer("""
#    SELECT * FROM building WHERE saw IN  ( SELECT general FROM lift )
#     """, return_tensors="pt")

# with torch.no_grad():

#     logits = model(**inputs).logits
# predicted_class_id = logits.argmax().item()
# print(predicted_class_id)

