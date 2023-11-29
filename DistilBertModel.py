from abc import ABC

from fine_tune_distilbert import tokenizer
from wafamole.models import Model
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch


class DistilBertModel(Model):

    def __int__(self):
        # self._tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        # self._model = DistilBertForSequenceClassification.from_pretrained("./sql_llm/sql_distilbert/")
        pass

    def extract_features(self, value: object):
        pass

    def classify(self, value: object):
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("./sql_llm/sql_distilbert/")
        inputs = tokenizer(value, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        return max(logits.numpy()[0])
