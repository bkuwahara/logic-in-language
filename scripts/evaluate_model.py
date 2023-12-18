import json
import os
import random
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
import csv
import argparse
#os.chdir("/w/339/bkuwahara/csc2542")
os.chdir("/w/246/ikozlov/csc2542-project")

random.seed(2023) # Needed to get same subset of questions each run (running all takes too long)

from models import LlamaBasic, LlamaLogical, RandModel

def load_data(dataset):
	#with open("./datasets/task.json") as data_file:
	with open(f"./datasets/{dataset}.json") as data_file:
		data = json.load(data_file)

	#prefix = data["task_prefix"]
	#prefix = "Identify the relation between the following premises and hypotheses, choosing from the options 'entailment' or 'non-entailment'.\n"
	questions = data["examples"]
	return questions




def evaluate(model_name, chain_of_thought, n_shots, path, dataset, output_dir, cache_dir='/w/339/bkuwahara/.cache/torch/kernels/', use_token=True):

	# Load in the model
	if model_name == "basic":
		model = LlamaBasic(path, chain_of_thought=chain_of_thought, n_shots=n_shots, cache_dir=cache_dir, use_token=use_token)
	elif model_name == "logical":
		model = LlamaLogical(path, chain_of_thought=chain_of_thought, n_shots=n_shots, cache_dir=cache_dir, use_token=use_token)
	elif model_name == "random":
		model = RandModel(path, chain_of_thought=chain_of_thought, n_shots=n_shots, cache_dir=cache_dir, use_token=use_token)
	else:
		raise ValueError("Model must be one of basic, logical, or random. Given {}".format(args.model))

	
	questions = load_data(dataset)
	#questions = random.choices(_questions, k=500) # Random sample of 500 questions to save time
	score = 0
	num_invalid = 0		


	# Set up output file for data
	savedir = f"./{output_dir}/{model_name}/{path}"
	if not os.path.exists(savedir):
		os.makedirs(savedir)
	output_file = f"{savedir}/{dataset}_{n_shots}shot"
	if chain_of_thought:
		output_file += "_cot"
	output_file += ".csv"
	
	with open(output_file, 'w', newline='') as output:
		writer = csv.writer(output)
		header = ["question_index", "response", "is_correct"]
		writer.writerow(header)

	# Loop through questions
	for i, q in tqdm(enumerate(questions[::])):
		query = q["input"]
		model_output = model(query, return_full_output=False)
		#print(i, model_output)
		is_invalid = model_output not in ["entailment", "non-entailment"]
		
		with open(output_file, 'a', newline='') as output:
			writer = csv.writer(output)
			correct = "NaN" if is_invalid else q["target_scores"][model_output]
			writer.writerow([i, model_output, correct])
		score += 0 if is_invalid else q["target_scores"][model_output]
		num_invalid += is_invalid		
	
	return score / len(questions), num_invalid



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
	parser.add_argument("--model", default="basic",
		help="The model to use. Options are basic, logical, or random")
	parser.add_argument("--path", default="meta-llama/Llama-2-13b-hf",
		help="The path to the LLM to use (huggingface path)")
	parser.add_argument("--output_dir", default="results",
		help="Directory to save results to")
	parser.add_argument("--n_shots", default=3, type=int,
		help="The number of examples the model will see in its prompt prior to inference (must have a pre-existing prompt file)")
	parser.add_argument("--chain_of_thought", default=0, type=int,
		help="Whether or not to use chain of thought reasoning")
	parser.add_argument("--dataset", default="task",
		help="Dataset on which to evaluate (task for basic epistemic, kd45_reasoning for kd45, mixed_reasoning for both")
	parser.add_argument("--cache_dir", default='/w/339/bkuwahara/.cache/torch/kernels/',
		help="Directory for storing cache files during program execution.")
	parser.add_argument("--use_token", default=True, type=bool,
		help="Whether or not to use token when calling model (necessary when calling Meta's LlaMa-2 model).")


	args = parser.parse_args()
	print(args.chain_of_thought)
#	print(args.model, args.output_dir, args.size)
	acc, inv = evaluate(args.model, args.chain_of_thought, args.n_shots, args.path, args.dataset, args.output_dir, cache_dir=args.cache_dir, use_token=args.use_token)
	
	print(acc,inv)



