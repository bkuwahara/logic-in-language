import json
import os
import random
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
os.chdir("/w/339/bkuwahara/csc2542")

from llama_basic import load_llama

def load_data():
	with open("./data/task.json") as data_file:
    		data = json.load(data_file)

	prefix = data["task_prefix"]
	questions = data["examples"]
	

def rand_model(query):
    return "entailment" if random.randint(0,1) == 0 else "non-entailment"


model, tokenizer = load_llama()


def evaluate(model, tokenizer):

    model_score=0
    for q in questions:
        query = q["input"]
        output = model(query)
        score = q["target_scores"][output]
        model_score += score

    return model_score / len(questions)



