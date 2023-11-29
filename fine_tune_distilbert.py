import csv
import collections
from datasets import Dataset, DatasetDict
import numpy as np
import evaluate
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

# Load sql injection data from csv into huggingface dataset.
def load_csv_to_ds():
    filename = "Modified_SQL_Dataset.csv"
    d = collections.defaultdict(list)
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k == "Label":
                    d["label"].append(int(v))
                else:
                    d[k].append(v)
    ds = Dataset.from_dict(d)
    ds_train_test = ds.train_test_split(test_size = 0.2, seed=42)
    test_valid = ds_train_test['test'].train_test_split(test_size=0.5, seed=42)
    ds_ttv = DatasetDict({
        'train': ds_train_test['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']
    })
    ds_ttv.save_to_disk("sql_ds.hf")
    return ds_ttv

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

#Helper function to tokenizer the dataset.
def tokenize_function(examples):
    return tokenizer(examples["Query"], padding="max_length", truncation=True)

#Maps the tokenize function to every sample in the dataset.
def tokenize_data(ds):
    tokenized_ds = ds.map(tokenize_function, batched=True)
    return tokenized_ds

metric = evaluate.load("accuracy")

#Helper Function to perform accuracy computations.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def fine_tune(token_ds):
    model_checkpoint = 'distilbert-base-uncased'
    id2label = {0: "Benign", 1: "Malicious"}
    label2id = {"Benign":0, "Malicious":1}
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=token_ds["train"],
        eval_dataset=token_ds["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model("sql_llm/sql_distilbert")
    trainer.evaluate(token_ds["test"])
    return model

def main():
    ds = load_csv_to_ds()
    token_ds = tokenize_data(ds)
    model = fine_tune(token_ds)
    return 0


if __name__ == "__main__":
    main()