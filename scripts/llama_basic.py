import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

llama_path = "/w/339/bkuwahara/llama_model/13b"




def load_llama():
	tokenizer = LlamaTokenizer.from_pretrained(llama_path, device_map="auto")
	model = LlamaForCausalLM.from_pretrained(llama_path, device_map="auto", offload_folder="offload", torch_dtype=torch.float16)
	#tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", device_map="auto")
	#model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", device_map="auto")
	#tokenizer = LlamaTokenizer.from_pretrained(llama_path)
	#model = LlamaForCausalLM.from_pretrained(llama_path)
	#device="cuda:0"
	#model.to(device)

	return model, tokenizer


if __name__ == "__main__":
	import json

	model, tokenizer = load_llama()
		

	# Test the model on the actual data
	with open("../datasets/task.json") as data_file:
    		data = json.load(data_file)

	prefix = data["task_prefix"]
	questions = data["examples"]

	q = questions[0]
	query = q["input"]
	prompt = prefix + query + "\nYour answer: "
	#print(prompt)
	input = tokenizer(prompt, return_tensors="pt").to("cuda")

	generate_ids = model.generate(**input, max_new_tokens=10)
	print("Reasoning prompt output: " + tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0])
