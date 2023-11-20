from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_mistral():
	tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
	tokenizer.pad_token = tokenizer.eos_token
	model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True)
	return model, tokenizer



#model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")


if __name__ == "__main__":
	import json
	model, tokenizer = load_mistral()

	test_prompt = "Hey, are you conscious? Can you talk to me?"
	inputs = tokenizer(test_prompt, return_tensors="pt")
	generate_ids = model.generate(**inputs, max_length=50)
	print("test prompt output: " + tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0])


	with open("../datasets/task.json") as data_file:
		data = json.load(data_file)

        #prefix = data["task_prefix"]
	prefix = "Decide if the following premise logically entails the hypothesis. Respond with \"entailment\" or \"non-entailment\""
	questions = data["examples"]
	q = questions[0]
	query = q["input"]
	input = tokenizer(prefix + query, return_tensors="pt").to("cuda")
	generate_ids = model.generate(**input, max_new_tokens=25)
	output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
	print(output)
