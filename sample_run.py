import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from DistilBertModel import DistilBertModel
from mutation_job import mutation_without_model, mutation_with_model

import sys, getopt

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model_sql = DistilBertForSequenceClassification.from_pretrained("./sql_llm/sql_distilbert/")
model_mutation = DistilBertForSequenceClassification.from_pretrained("./sql_llm/mutation_sql_distilbert")


def main(argv):
    query = "admin' OR 1=1#"
    max_rounds = 500
    round_size = 20
    opts, args = getopt.getopt(argv, "hq:r:", ["query=", "rounds="])
    for opt, arg in opts:
        if opt == '-h':
            print('sample_run.py -q "<query>" -r <max_rounds)>')
            sys.exit()
        elif opt in ("-q", "--query"):
            query = arg
        elif opt in ("-r", "--rounds"):
            max_rounds = arg
    print('original query is ', query)
    print('max rounds is  ', max_rounds)
    m_model = DistilBertModel()

    inputs1 = tokenizer(query, return_tensors="pt")

    min_confidence, mutated_query = mutation_with_model(query, round_size, 100, m_model)

    with torch.no_grad():
        logits = model(**inputs1).logits
        sql_logits = model_sql(**inputs1).logits
        mutation_logits = model_mutation(**inputs1).logits

    model_label = logits.argmax().item()
    sql_logits_label = sql_logits.argmax().item()
    mutation_logits_label = mutation_logits.argmax().item()

    print(
        f'original query  : model_label: {model_label}, sql_logits_label: {sql_logits_label}, mutation_logits_label: {mutation_logits_label}')

    inputs1 = tokenizer(mutated_query, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs1).logits
        sql_logits = model_sql(**inputs1).logits
        mutation_logits = model_mutation(**inputs1).logits

    model_label = logits.argmax().item()
    sql_logits_label = sql_logits.argmax().item()
    mutation_logits_label = mutation_logits.argmax().item()
    print(
        f'mutated query  : model_label: {model_label}, sql_logits_label: {sql_logits_label}, mutation_logits_label: {mutation_logits_label}')



if __name__ == "__main__":
    main(sys.argv[1:])
