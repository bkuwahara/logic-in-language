import torch
import random
from transformers import LlamaForCausalLM, LlamaTokenizer
from epistemic_logic import KnowledgeBase

llama_path = "meta-llama"

"""
Class for doing inference directly with the model
"""
class LlamaBasic:
	

	# model: string specifying the model to use, e.g. "13b"
	def __init__(self, model):
		
		load_from = f"{llama_path}/Llama-2-{model}"
		tokenizer = LlamaTokenizer.from_pretrained(load_from)
		model = LlamaForCausalLM.from_pretrained(load_from)
		self.model = model
		self.tokenizer = tokenizer

	# prompt: string giving the task prompt for the model to perform inference on
	def __call__(self, prompt, max_new_tokens=10):
		input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
		generate_ids = self.model.generate(**input, max_new_tokens=max_new_tokens)
		model_output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
		return model_output

"""
Class for wrapping an LLM with symbolic epistemic reasoning system
"""
class LlamaLogical:
	conversion_prompt = "!!!TO DO!!!"

	# model: string specifying the model to use, e.g. "13b"
	def __init__(self, model):		
		load_from = f"{llama_path}/Llama-2-{model}"
		tokenizer = LlamaTokenizer.from_pretrained(load_from)
		model = LlamaForCausalLM.from_pretrained(load_from, **kwargs)
		self.model = model
		self.tokenizer = tokenizer


	# Prompts the model to convert a statement into epistemic logic in string form
	# statement: string to convert into logic
	def convert_to_logic(self, statement):
		input = self.tokenizer(LlamaLogical.conversion_prompt+statement, return_tensors="pt").to("cuda")
		generate_ids = self.model.generate(**input, max_new_tokens=max_new_tokens)
		model_output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
		# TO DO: Write regex script to extract the model's answer from output
		logic_string = None #REPLACE THIS

		return logic_string


	# Gets the model's prediction for a given problem
	# task: string specifying the inference task
	def __call__(self, task):
		prem, hyp = prompt.split("Premise: ", maxsplit=1)[1].split("Hypothesis: ")

		premise_logic = self.convert_to_logic(prem)
		hypothesis_logic = self.convert_to_logic(hyp)

		KB_prem = KnoweledgeBase.from_string(premise_logic)
		KB_hyp = KnowledgeBase.from_string(hypothesis_logic)
		
		answer = KB_prem.entails(KB_hyp)
		return "entailment" if answer else "non-entailment"


# For testing the rest of the code quickly
class RandModel:
	def __init__(self, model):
		pass

	# Randomly returns an output for any prompt
	# returns prompt as well to mimic LLM behaviour
	def __call__(self, prompt):
		bit = random.randint(0,2)
		if bit == 0:
			return prompt + "entailment"
		elif bit == 1:
			return prompt + "non-entailment"
		else:
			return prompt + "I don't understand"

if __name__ == "__main__":
	import json

	model = LlamaBasic("13b")
		

	# Test the model on the actual data
	with open("../datasets/task.json") as data_file:
    		data = json.load(data_file)

	prefix = data["task_prefix"]
	questions = data["examples"]

	q = questions[0]
	query = q["input"]
	prompt = prefix + query + "\nYour answer: "
	#print(prompt)

	out = model(prompt)
	print("Reasoning prompt output: " + out)
