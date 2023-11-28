from DistilBertModel import DistilBertModel
from fine_tune_distilbert import load_csv_to_ds
from wafamole.evasion import EvasionEngine
from wafamole.payloadfuzzer.sqlfuzzer import SqlFuzzer
import csv

def mutation_without_model(payload, round_size):
    fuzzer = SqlFuzzer(payload)
    payloads = {fuzzer.fuzz() for _ in range(round_size)}
    return payloads


max_rounds = 1000
round_size = 20
threshold = 0.5
timeout = 14400
# mutationModel = DistilBertModel()

# engine = EvasionEngine(mutationModel)
ds = load_csv_to_ds()
result = []
count = 0
for sets in ds:
    for item in ds[sets]:
        count += 1
        payloads = mutation_without_model(item['Query'], round_size)
        res = [{'Query': i, 'label': item["label"]} for i in payloads]
        result += res
print(f'total Query {count}')
# print(result)
# payloads = mutation_without_model(ds["train"][0]["Query"], round_size)
# res = [{'Query': item, 'label': ds["train"][0]["label"]} for item in payloads]
# print(res)
# for item in ds["train"]:
#     print(item["Query"])
query_body = """"admin' OR 1=1#"""
with open('./mutation_SQL_datasets', 'w', newline='') as csvfile:
    fieldnames = ['Query', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Write the data
    for row in result:
        writer.writerow(row)
# payloads = mutation_without_model(query_body, round_size)
# min_confidence, min_payload = engine.evaluate(query_body, max_rounds, round_size, timeout, threshold)
# print(payloads)
