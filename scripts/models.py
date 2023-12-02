import torch
import random
from transformers import LlamaForCausalLM, LlamaTokenizer
from epistemic_logic import KnowledgeBase

llama_path = "meta-llama"


class LlamaBasic:
	def __init__(self, size):
		
		load_from = f"{llama_path}/Llama-2-{size}"
		tokenizer = LlamaTokenizer.from_pretrained(load_from)
		model = LlamaForCausalLM.from_pretrained(load_from)
		self.model = model
		self.tokenizer = tokenizer

	def __call__(self, prompt, max_new_tokens=10):
		input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
		generate_ids = self.model.generate(**input, max_new_tokens=max_new_tokens)
		model_output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
		return model_output


class LlamaLogical:
	conversion_prompt = "!!!TO DO!!!"

	def __init__(self, size):		
		load_from = f"{llama_path}/Llama-2-{size}"
		tokenizer = LlamaTokenizer.from_pretrained(load_from)
		model = LlamaForCausalLM.from_pretrained(load_from, **kwargs)
		self.model = model
		self.tokenizer = tokenizer


	def convert_to_logic(self, statement):
		input = self.tokenizer(LlamaLogical.conversion_prompt+statement, return_tensors="pt").to("cuda")
		generate_ids = self.model.generate(**input, max_new_tokens=max_new_tokens)
		model_output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
		# TO DO: Write regex script to extract the model's answer from output
		logic_string = None #REPLACE THIS

		return logic_string


	def __call__(self, task):
		# Only pseudo-code for now!
		# prem_formula, hyp_formula = to_logic(question)
		# ans = is_entailment(prem_formula, hyp_formula)
		# return ans

		# TO DO: write regex or split code to extract premise and hypothesis from question
		premise_str = None #REPLACE THIS
		hypothesis_str = None #REPLACE THIS

		premise_logic = self.convert_to_logic(premise_str)
		hypothesis_logic = self.convert_to_logic(hypothesis_str)

		KB_prem = KnoweledgeBase.from_string(premise_logic)
		KB_hyp = KnowledgeBase.from_string(hypothesis_logic)
		
		answer = KB_prem.entails(KB_hyp)
		return "entailment" if answer else "non-entailment"


# For testing the rest of the code quickly
class RandModel:
	def __init__(self, size):
		pass


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
