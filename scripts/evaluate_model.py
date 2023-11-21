import json
import os
import random
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import tqdm
os.chdir("/w/339/bkuwahara/csc2542")

from llama_basic import load_llama

def load_data():
	with open("./datasets/task.json") as data_file:
		data = json.load(data_file)

	prefix = data["task_prefix"]
	questions = data["examples"]
	return prefix, questions

def rand_model(query):
	return "entailment" if random.randint(0,1) == 0 else "non-entailment"



def evaluate(model, tokenizer):

	prefix, questions = load_data()
	model_score = 0
	invalid = 0 # responses that don't fit the desired template
	for q in tqdm(questions):
		query = q["input"]
		prompt = prefix + query + "\nYour answer: "
		input = tokenizer(prompt, return_tensors="pt").to("cuda")
		generate_ids = model.generate(**input, max_new_tokens=10)
		model_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
		ans = model_output.split("Your answer:")[1].split()[0]
		if ans not in ["entailment", "non-entailment"]:
			invalid += 1
		score = q["target_scores"][ans]
		model_score += score


	return model_score / len(questions), invalid



if __name__ == "__main__":
	model, tokenizer = load_llama()
	acc, inv = evaluate(model, tokenizer)
	
	with open("./results/llama_basic.txt", "a") as outfile:
		outfile.write("Accuracy: {}\nInvalid responses: {}".format(acc, inv))




