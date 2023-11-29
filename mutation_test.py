import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm
from datasets import load_from_disk

from mutation_job import mutation_without_model

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("./sql_llm/sql_distilbert/")

arrow_datasets_reloaded = load_from_disk("sql_ds.hf")
test_set = arrow_datasets_reloaded["test"]


def predict(query):
    inputs = tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.argmax().item()

query = mutation_without_model('SELECT COUNT ( ProductID )  AS NumberOfProducts FROM Products;',1)
print(query)
correct_classify = 0
misclassify_after_mutation = 0
for item in tqdm(test_set):
    pred_y = predict(item['Query'])
    mutated_query = mutation_without_model(item['Query'], 1)
    pred_after_mutation = predict(list(mutated_query)[0])
    if pred_y == item['label']:
        correct_classify += 1
        if pred_y != pred_after_mutation:
            misclassify_after_mutation += 1
print(f'correct classify {correct_classify}')
print(f'miss classify {misclassify_after_mutation}')
print(f'success rate {misclassify_after_mutation / correct_classify}')
