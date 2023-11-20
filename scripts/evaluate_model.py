import json
import os
import random


os.chdir("/w/339/bkuwahara/csc2542")

with open("./data/task.json") as data_file:
    data = json.load(data_file)

prefix = data["task_prefix"]
questions = data["examples"]

def rand_model(query):
    return "entailment" if random.randint(0,1) == 0 else "non-entailment"


def llama_2_unassisted(input, k):
    pass

def llama_2_logical(input, k):
    pass


def evaluate(model):

    model_score=0
    for q in questions:
        query = q["input"]
        output = model(query)
        score = q["target_scores"][output]
        model_score += score

    return model_score / len(questions)
    


