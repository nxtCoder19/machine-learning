from pprint import pprint
import ray
from datasets import load_dataset
from transformers import AutoTokenizer
import ray.data

ray.init()
pprint(ray.cluster_resources())

use_gpu = True  # set this to False to run on CPUs
num_workers = 1  # set this to number of GPUs or CPUs you want to use

GLUE_TASKS = [
    "cola",
    "mnli",
    "mnli-mm",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
]
task = "cola"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

actual_task = "mnli" if task == "mnli-mm" else task
datasets = load_dataset("glue", actual_task)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
ray_datasets = {
    "train": ray.data.from_huggingface(datasets["train"]),
    "validation": ray.data.from_huggingface(datasets["validation"]),
    "test": ray.data.from_huggingface(datasets["test"]),
}
pprint(ray_datasets)
