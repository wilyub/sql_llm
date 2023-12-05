import os
from multiprocessing import freeze_support

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from DistilBertModel import DistilBertModel
from fine_tune_distilbert import load_csv_to_ds
from wafamole.evasion import EvasionEngine
from wafamole.payloadfuzzer.sqlfuzzer import SqlFuzzer
from concurrent.futures import ThreadPoolExecutor
import csv


def mutation_without_model(payload, round_size):
    fuzzer = SqlFuzzer(payload)

    payloads = fuzzer.set_fuzz()
    # payloads = {fuzzer.fuzz() for _ in range(round_size)}
    return payloads


def _mutation_round(payload, round_size, model):
    fuzzer = SqlFuzzer(payload)
    # pool = ThreadPoolExecutor(max_workers=2)
    # Some mutations do not apply to some payloads
    # This removes duplicate payloads
    payloads = fuzzer.set_fuzz()
    # {fuzzer.fuzz() for _ in range(round_size)}
    results = map(model.classify, payloads)
    confidence, payload = min(zip(results, payloads))
    # pool.shutdown()
    return confidence, payload


max_rounds = 1000
round_size = 20
threshold = 0.5
timeout = 14400


#
# ds = load_csv_to_ds()
# result = []
# count = 0
# for sets in ds:
#     for item in ds[sets]:
#         count += 1
#         payloads = mutation_without_model(item['Query'], round_size)
#         res = [{'Query': i, 'label': item["label"]} for i in payloads]
#         result += res
# print(f'total Query {count}')
# query_body = """"admin' OR 1=1#"""
# with open('./mutation_without_model_SQL_datasets', 'w', newline='') as csvfile:
#     fieldnames = ['Query', 'label']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#     # Write the header
#     writer.writeheader()
#
#     # Write the data
#     for row in result:
#         writer.writerow(row)


# fuzzer = SqlFuzzer(query)
# payloads = {fuzzer.fuzz() for _ in range(round_size)}
# results = map(mutationModel.classify, payloads)
# print(results)
# abc = zip(results, payloads)
# print(abc)
# confidence, payload = min(zip(results, payloads))
# print(confidence)
# print(payload)

def mutation_with_model(query_body, round_size, max_rounds, mutationModel):
    evaluation_results = []
    min_confidence, min_payload = _mutation_round(query_body, round_size, mutationModel)
    # min_confidence, min_payload = engine.evaluate(query_body, max_rounds, round_size, timeout, threshold)
    evaluation_results.append((min_confidence, min_payload))
    while max_rounds > 0 and min_confidence > threshold:
        for candidate_confidence, candidate_payload in sorted(
                evaluation_results
        ):
            max_rounds -= 1

            confidence, payload = _mutation_round(
                candidate_payload, round_size, mutationModel
            )
            if confidence < candidate_confidence:
                evaluation_results.append((confidence, payload))
                min_confidence, min_payload = min(evaluation_results)
                break

    if min_confidence < threshold:
        print("[+] Threshold reached")
    elif max_rounds <= 0:
        print("[!] Max number of iterations reached")

    print(
        "Reached confidence {}\nwith payload\n{}\n round left {}".format(
            min_confidence, min_payload, max_rounds
        )
    )
    return min_confidence, min_payload
    # print(min_payload)
    # print(min_confidence)

# mutation with model prediction
# mutationModel = DistilBertModel()
# engine = EvasionEngine(mutationModel)
# ds = load_csv_to_ds()
# result = []
# count = 0
# min_confidence, min_payload = mutation_with_model("""
# "sELECt canNOT frOM/**/HaLf UniOn select CuRVE fRom VeRy ||DEr by maCHiNerY"
# """, round_size, max_rounds, mutationModel)
# print(min_payload)
# print(min_confidence)
# for sets in ds:
#     for item in ds[sets]:
#         count += 1
#         min_confidence, min_payload = mutation_with_model(item['Query'], round_size,max_rounds, mutationModel)
#         res = {'Query': min_payload, 'label': item["label"]}
#         result.append(res)
# print(count)
# with open('./mutation_with_model_SQL_datasets_train.csv', 'w', newline='') as csvfile:
#     fieldnames = ['Query', 'label']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#     # Write the header
#     writer.writeheader()
#
#     # Write the data
#     for row in result:
#         writer.writerow(row)
