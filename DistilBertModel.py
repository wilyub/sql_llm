from abc import ABC

from fine_tune_distilbert import tokenizer
from wafamole.models import Model
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from torch.nn import Sigmoid
import torch


class DistilBertModel(Model):
    _model = DistilBertForSequenceClassification.from_pretrained("./sql_llm/mutation_sql_distilbert/")
    _tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def __int__(self):
        # super(DistilBertModel, self).__init__()
        # self._tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        # self._model = DistilBertForSequenceClassification.from_pretrained("./sql_llm/sql_distilbert/")
        pass

    def extract_features(self, value: object):
        pass

    def classify(self, value: object):
        # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        # model = DistilBertForSequenceClassification.from_pretrained("./sql_llm/sql_distilbert/")
        inputs = self._tokenizer(value, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**inputs).logits

        sig = Sigmoid()
        prob = sig(logits)
        # print(prob.numpy()[0])
        return max(prob.numpy()[0])

    def predict(self, value: object):
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("./sql_llm/sql_distilbert/")
        inputs = tokenizer(value, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        return logits.argmax().item()
