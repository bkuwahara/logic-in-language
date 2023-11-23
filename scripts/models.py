import torch
import random
from transformers import LlamaForCausalLM, LlamaTokenizer

llama_path = "/w/339/bkuwahara/llama_model/"


class LlamaBasic:
	def __init__(self, size, local=True):
		
		load_from = f"{llama_path}/{size}" if local else f"meta-llama/Llama-2-{size}-chat-hf"
		kwargs = dict() if local else {"device_map" : "auto", "offload_folder": "offload", "torch_dtype" : torch.float16}
		tokenizer = LlamaTokenizer.from_pretrained(load_from, device_map="auto" if local else None)
		model = LlamaForCausalLM.from_pretrained(load_from, **kwargs)
		self.model = model
		self.tokenizer = tokenizer

	def __call__(self, prompt, max_new_tokens=10):
		input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
		generate_ids = self.model.generate(**input, max_new_tokens=max_new_tokens)
		model_output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
		return model_output


class LlamaLogical:
	def __init__(self, size):		
		load_from = f"{llama_path}/{size}" if local else f"meta-llama/Llama-2-{size}-chat-hf"
		kwargs = dict() if local else {"device_map" : "auto", "offload_folder": "offload", "torch_dtype" : torch.float16}
		tokenizer = LlamaTokenizer.from_pretrained(load_from, device_map="auto" if local else None)
		model = LlamaForCausalLM.from_pretrained(load_from, **kwargs)
		self.model = model
		self.tokenizer = tokenizer


	def parse_question(self, prompt):
		premise, hypothesis = prompt.split("Premise: ")[1].split("Hypothesis: ")
		return premise, hypothesis

	def extract_formulas(self, raw_output):
		# To add later
		pass

	def to_logic(question):
		premise, hypothesis = self.parse_question(question)
		prefix = """
			Translate the following statement into epistemic logic, 
			using K(a, x) to denote that a knows x and B(a, x) to 
			denote that a believes x: 
			"""

		inputs = tokenizer([prefix+premise, prefix+hypothesis], return_tensors="pt").to("cuda")
		generate_ids = model.generate(**input, max_new_tokens=100)
		outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
		formulas = self.extract_formulas(outputs)
		return formulas


	def __call__(self, task):
		# Only pseudo-code for now!
		# prem_formula, hyp_formula = to_logic(question)
		# ans = is_entailment(prem_formula, hyp_formula)
		# return ans

		pass

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
