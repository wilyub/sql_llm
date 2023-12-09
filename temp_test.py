import csv

import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm
from datasets import load_from_disk
import random

from DistilBertModel import DistilBertModel
from mutation_job import mutation_without_model, mutation_with_model

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model_sql = DistilBertForSequenceClassification.from_pretrained("./sql_llm/sql_distilbert/")
model_mutation = DistilBertForSequenceClassification.from_pretrained("./sql_llm/mutation_sql_distilbert")

query1 = "SELECT option_value FROM wp_options WHERE option_name  =  'zcf_captcha_settings' LIMIT 1"
query2 = """
SELECT/*(SELECT (SELECT 11)) AND "s)qj" LIKE "s)qj" AND (SELECT 1)#t>(Y50*/option_value/**/FROM/**/wp_options/*+S|=_TE/*/WHERE/*%N*/option_name  LIKE  'zcf_captcha_settings' LIMIT/**/1%Yq
"""
inputs1 = tokenizer(query1, return_tensors="pt")
inputs2 = tokenizer(query2, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs1).logits
    sql_logits = model_sql(**inputs1).logits
    mutation_logits = model_mutation(**inputs1).logits

with torch.no_grad():
    logits2 = model(**inputs2).logits
    sql_logits2 = model_sql(**inputs2).logits
    mutation_logits2 = model_mutation(**inputs2).logits

model_label = logits.argmax().item()
sql_logits_label = sql_logits.argmax().item()
mutation_logits_label = mutation_logits.argmax().item()

model_label2 = logits2.argmax().item()
sql_logits_label2 = sql_logits2.argmax().item()
mutation_logits_label2 = mutation_logits2.argmax().item()

print(
    f'query1 : model_label: {model_label}, sql_logits_label: {sql_logits_label}, mutation_logits_label: {mutation_logits_label}')
print(
    f'mutated query : model_label: {model_label2}, sql_logits_label: {sql_logits_label2}, mutation_logits_label: {mutation_logits_label2}')
