# Building Better WAF Models with Transformers

## Overview

SQL Injections are a common attack method for tricking a system into executing unintended actions by embedding malicious code in a SQL query. Modern systems employ defense mechanisms, such as "Web Application Firewalls" (WAF), to protect against these attacks. However, attackers have developed techniques to evade WAFs, as highlighted in the paper "WAF-A-MoLE: Evading Web Application Firewalls through Adversarial Machine Learning."

This repository presents our work aimed at improving existing WAF models to better defend against SQL injection attacks. We utilize fine-tuning on a DistilBert model, demonstrating high accuracy in detecting mutated malicious queries.

## Repository Structure

- `evasion/`: Contains the code for mutation.
- `sql_llm/`: Includes models for training: both original and mutated ones.
- `wafamole/`: Code provided by the paper.
- `DistilBERTModel.py/`: Code for Distil BERT.
- `fine_tune_distilbert.py`: Code for finetuning distil BERT on sql injection dataset.
- `fine_tune_mutation_distilbert.py`:  Code for finetuning distil BERT on mutated sql injection dataset.
- `testoutput/`: Results.

## Getting Started

### Prerequisites

- Python 3.x
- numpy
- keras
- scikit-learn
- joblib
- sqlparse
- networkx
- click
- tensorflow
- datasets
- transformers
- torch
- h5py
- tqdm
- evaluate

## Usage

Run either `fine_tune_distilbert.py` or `fine_tune_mutation_distilbert.py` for tweaking the finetuning process on both versions, with and without the mutated data. You can also run (tweak) `test_sql_distilbert.py` to test the model(s).

