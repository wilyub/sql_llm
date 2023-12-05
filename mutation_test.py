import csv

import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm
from datasets import load_from_disk
import random

from DistilBertModel import DistilBertModel
from mutation_job import mutation_without_model, mutation_with_model

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("./sql_llm/sql_distilbert/")

arrow_datasets_reloaded = load_from_disk("sql_ds.hf")
test_set = arrow_datasets_reloaded["test"]


def predict(query):
    inputs = tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.argmax().item()


# query = 'SELECT COUNT ( ProductID )  AS NumberOfProducts FROM Products;'

correct_classify = 0
misclassify_after_mutation = 0
# print(query)
# for i in tqdm(range(4)):
#     query = mutation_without_model(query, 20)
#     query = random.choice(list(query))
#     print(query)
# mutation_round = 20
# for item in tqdm(test_set):
#     query = item['Query']
#     pred_y = predict(query)
#     if pred_y == item['label']:
#         correct_classify += 1
#         for i in range(mutation_round):
#             query = mutation_without_model(query, 20)
#             query = random.choice(list(query))
#         pred_after_mutation = predict(query)
#         if pred_y != pred_after_mutation:
#             misclassify_after_mutation += 1
# print(f'correct classify {correct_classify}')
# print(f'miss classify {misclassify_after_mutation}')
# print(f'success rate {misclassify_after_mutation / correct_classify}')

#### mulitple round mutation with model
count = 0
m_model = DistilBertModel()
max_rounds = 1000
round_size = 20
correct_classify = 0
misclassify_after_mutation = 0
result = []
for item in tqdm(test_set):
    if count == 20:
        break
    count += 1
    pred_y = predict(item['Query'])
    print(item['Query'])
    min_confidence, mutated_query = mutation_with_model(item['Query'], round_size, max_rounds, m_model)
    pred_after_mutation = predict(list(mutated_query)[0])
    res = {'Query': mutated_query, 'label': item['label']}
    result.append(res)
    if pred_y == item['label']:
        correct_classify += 1
        if pred_y != pred_after_mutation:
            misclassify_after_mutation += 1
print(f'correct classify {correct_classify}')
print(f'miss classify {misclassify_after_mutation}')
print(f'success rate {misclassify_after_mutation / correct_classify}')

with open('./mutation_with_model_SQL_datasets', 'w', newline='') as csvfile:
    fieldnames = ['Query', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Write the data
    for row in result:
        writer.writerow(row)
