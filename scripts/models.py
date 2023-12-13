import torch
import random
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from epistemic_logic import KnowledgeBase, is_entailment

#os.chdir("/w/339/bkuwahara/csc2542")
os.chdir("/w/246/ikozlov/csc2542-project")

"""
Class for doing inference directly with the model
"""

token = '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'

class LlamaBasic:
	# model: string specifying the model to use, e.g. "13b"
	def __init__(self, model_path, n_shots=3, chain_of_thought=False, cache_dir='/w/339/bkuwahara/.cache/torch/kernels/', use_token=True):

		self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		offload_state_dict = torch.cuda.is_available() 
		
		if use_token: 
			tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, token=token)
			model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',offload_folder='offload', offload_state_dict=offload_state_dict, cache_dir=cache_dir, token=token) 
		else: 
			tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
			model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',offload_folder='offload', offload_state_dict=offload_state_dict, cache_dir=cache_dir)
			
		self.model = model
		self.tokenizer = tokenizer
		self.chain_of_thought = chain_of_thought

		
		prompt_name = f"respond_{n_shots}shot" + ("_cot" if chain_of_thought else "")
		with open(f"./prompts/{prompt_name}.txt", 'r') as f:
			prompt = f.read()
			self.prompt=prompt


		prompt_acts = f"./prompts/{model_path}/{prompt_name}.pt"
		if os.path.isfile(prompt_acts):
			self.encoded_prompt = torch.load(prompt_acts)
		else:
			directory = f"./prompts/{model_path}"
			if not os.path.exists(directory):
				os.makedirs(directory)
			input_ids = tokenizer.encode(self.prompt, return_tensors='pt').to(self.device)
			outputs = model(input_ids, output_hidden_states=True)
			self.encoded_prompt = outputs.past_key_values
			torch.save(self.encoded_prompt, prompt_acts)



	# prompt: string giving the task prompt for the model to perform inference on
	def __call__(self, task, return_full_output=False):
		max_new_tokens = 100 if self.chain_of_thought else 8
		input = self.tokenizer(self.prompt+'\n'+task, return_tensors="pt").to(self.device)
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
	def __init__(self, model_path, n_shots=3, chain_of_thought=False, cache_dir='/w/339/bkuwahara/.cache/torch/kernels/', use_token=True):		
		self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		offload_state_dict = torch.cuda.is_available()
		
		if use_token: 
			tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, token=token)
			model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',offload_folder='offload', offload_state_dict=offload_state_dict, cache_dir=cache_dir, token=token)
		else:
			tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
			model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',offload_folder='offload', offload_state_dict=offload_state_dict, cache_dir=cache_dir)
		
		self.model = model
		self.tokenizer = tokenizer
		self.chain_of_thought=chain_of_thought

		prompt_name = f"translate_{n_shots}shot" + ("_cot" if chain_of_thought else "")
		with open(f"./prompts/{prompt_name}.txt", 'r') as f:
			prompt = f.read()
			self.prompt=prompt


		prompt_acts = f"./prompts/{model_path}/{prompt_name}.pt"
		if os.path.isfile(prompt_acts):
			self.encoded_prompt = torch.load(prompt_acts)
		else:
			input_ids = tokenizer.encode(self.prompt, return_tensors='pt').to(self.device)
			outputs = model(input_ids, output_hidden_states=True)
			self.encoded_prompt = outputs.past_key_values
			torch.save(self.encoded_prompt, prompt_acts)


	# Gets the model's prediction for a given problem
	# task: string specifying the inference task
	def __call__(self, task, return_full_output=False):
		max_new_tokens = 150
		input = self.tokenizer(self.prompt+'\n'+task, return_tensors="pt").to(self.device)
		generate_ids = self.model.generate(**input, max_new_tokens=max_new_tokens, past_key_values=self.encoded_prompt)
		model_output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

		if return_full_output:
			return model_output
		else:
			try:
				_p, _h = model_output.split("Translated premise: ")[-1].split("Translated hypothesis: ")
				premise_logic = _p.split("$$")[1]
				hypothesis_logic = _h.split("$$")[1]

				#KB_prem = KnowledgeBase.from_string(premise_logic)
				#KB_hyp = KnowledgeBase.from_string(hypothesis_logic)
				#answer = KB_prem.entails(KB_hyp)
				#return "entailment" if answer else "non-entailment"
				answer = is_entailment(premise_logic,hypothesis_logic)
				if answer != 'entailment':
					return 'non-entailment' # for now both inconsistency and non-entailment should be treated as non-entailment
				return answer
			except:
				return model_output[len(self.prompt):]



#		prem, hyp = task.split("Premise: ", maxsplit=1)[1].split("Hypothesis: ")
#		print("Premise string: " + prem)
#		print("Hypothesis string: " + hyp)
#		premise_logic = self.convert_to_logic(prem, max_new_tokens=max_new_tokens)
#		hypothesis_logic = self.convert_to_logic(hyp, max_new_tokens=max_new_tokens)
#		print("premise logic: " + premise_logic)
#		print("hypothesis logic: " + hypothesis_logic)
#


# For testing the rest of the code quickly
class RandModel:
	def __init__(self, model_path, **kwargs):
		pass

	# Randomly returns an output for any prompt
	def __call__(self, prompt, **kwargs):
		bit = random.randint(0,2)
		if bit == 0:
			return "entailment"
		elif bit == 1:
			return "non-entailment"
		else:
			return "I don't understand"

if __name__ == "__main__":
	import json
#	path="meta-llama/Llama-2-13b-hf"
	path="TheBloke/llemma_34b-AWQ"
#	model = LlamaBasic(path)
#	model = LlamaLogical(path)
#	model = LlamaBasic(path, chain_of_thought=True, n_shots=3)
	model = LlamaBasic(path, chain_of_thought=True, n_shots=3, cache_dir='/w/246/ikozlov/.cache/torch/kernels/', use_token=False)
#	model = LlamaLogical(path, chain_of_thought=True)

#	model = LlamaLogical("meta-llama/Llama-2-13b-hf")

	# Test the model on the actual data
	with open("./datasets/task.json") as data_file:
		data = json.load(data_file)

	#prefix = data["task_prefix"]
	questions = random.choices(data["examples"], k=1)
	
	for question in questions:
		q = question["input"]
	#	print(q)

		#out = model.convert_to_logic(q,max_new_tokens=100, return_full_output=True)
	#print("Full model output: "+out)
	#logic_string = out.split("Answer: ")[-1].split("$$")[1]
	#print("Isolated logic string: "+logic_string)
		answer = model(q, return_full_output=False)
		print(answer)