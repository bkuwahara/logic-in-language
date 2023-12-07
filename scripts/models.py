import torch
import random
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from epistemic_logic import KnowledgeBase

os.chdir("/w/339/bkuwahara/csc2542")

"""
Class for doing inference directly with the model
"""

class LlamaBasic:
	# model: string specifying the model to use, e.g. "13b"
	def __init__(self, model_path):

		self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		
		tokenizer = AutoTokenizer.from_pretrained(model_path)
		model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',offload_folder='offload')
		self.model = model
		self.tokenizer = tokenizer

		with open("./prompts/prompt_to_answer.txt", 'r') as f:
			prompt = f.read()
			self.prompt=prompt


		prompt_acts = f"./prompts/{model_path}/respond.pt"
		if os.path.isfile(prompt_acts):
			self.encoded_prompt = torch.load(prompt_acts)
		else:
			input_ids = tokenizer.encode(self.prompt, return_tensors='pt').to(self.device)
			outputs = model(input_ids, output_hidden_states=True)
			self.encoded_prompt = outputs.past_key_values
			torch.save(self.encoded_prompt, prompt_acts)



	# prompt: string giving the task prompt for the model to perform inference on
	def __call__(self, prompt, max_new_tokens=100, return_full_output=False):
		input = self.tokenizer(self.prompt+'\n'+prompt, return_tensors="pt").to(self.device)
		generate_ids = self.model.generate(**input, max_new_tokens=max_new_tokens, past_key_values=self.encoded_prompt)
		model_output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
		if not return_full_output:
			answer = model_output.split("Answer: ")[-1].split()[0]
			return answer
		else:
			return model_output

"""
Class for wrapping an LLM with symbolic epistemic reasoning system
"""
class LlamaLogical:

	# model: string specifying the model to use, e.g. "13b"
	def __init__(self, model_path):		
		self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		
		tokenizer = AutoTokenizer.from_pretrained(model_path)
		model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',offload_folder='offload')
		self.model = model
		self.tokenizer = tokenizer

		with open("./prompts/prompt_to_translate.txt", 'r') as f:
			prompt = f.read()
			self.prompt=prompt


		prompt_acts = f"./prompts/{model_path}/translate.pt"
		if os.path.isfile(prompt_acts):
			self.encoded_prompt = torch.load(prompt_acts)
		else:
			input_ids = tokenizer.encode(self.prompt, return_tensors='pt').to(self.device)
			outputs = model(input_ids, output_hidden_states=True)
			self.encoded_prompt = outputs.past_key_values
			torch.save(self.encoded_prompt, prompt_acts)


	# Prompts the model to convert a statement into epistemic logic in string form
	# statement: string to convert into logic
	def convert_to_logic(self, statement, max_new_tokens=100, return_full_output=False):
		input = self.tokenizer(self.prompt+'\n'+"Premise: "+statement, return_tensors="pt").to(self.device)
		generate_ids = self.model.generate(**input, max_new_tokens=max_new_tokens, past_key_values=self.encoded_prompt)
		
		
		model_output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

		if return_full_output:
			return model_output
		else:
			logic_string = model_output.split("Answer: ")[-1].split("$$")[1]
			return logic_string


	# Gets the model's prediction for a given problem
	# task: string specifying the inference task
	def __call__(self, task):
		prem, hyp = task.split("Premise: ", maxsplit=1)[1].split("Hypothesis: ")
		print("Premise string: " + prem)
		print("Hypothesis string: " + hyp)
		premise_logic = self.convert_to_logic(prem)
		hypothesis_logic = self.convert_to_logic(hyp)
		print("premise logic: " + premise_logic)
		print("hypothesis logic: " + hypothesis_logic)
		KB_prem = KnowledgeBase.from_string(premise_logic)
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

	model = LlamaBasic("meta-llama/Llama-2-13b-hf")
			
#	model = LlamaLogical("meta-llama/Llama-2-13b-hf")

	# Test the model on the actual data
	with open("./datasets/task.json") as data_file:
    		data = json.load(data_file)

	#prefix = data["task_prefix"]
	questions = data["examples"]

	q = questions[0]["input"]
	print(q)

	#q = "Joseph suspects that Charles believes that a white dog is running through the water at a beach."

	#out = model.convert_to_logic(q,max_new_tokens=100, return_full_output=True)
	#print("Full model output: "+out)
	#logic_string = out.split("Answer: ")[-1].split("$$")[1]
	#print("Isolated logic string: "+logic_string)
	answer = model(q, return_full_output=True)
	print(answer)
