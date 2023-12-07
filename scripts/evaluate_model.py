import json
import os
import random
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
import csv
import argparse
import tqdm
os.chdir("/w/339/bkuwahara/csc2542")

from models import LlamaBasic, LlamaLogical, RandModel

def load_data():
	with open("./datasets/task.json") as data_file:
		data = json.load(data_file)

	prefix = data["task_prefix"]
	questions = data["examples"]
	return prefix, questions




def evaluate(model_name, path, output_dir):

	# Load in the model
	if model_name == "basic":
		model = LlamaBasic(path)
	elif model_name == "logical":
		model = LlamaLogical(path)
	elif model_name == "random":
		model = RandModel(path)
	else:
		raise ValueError("Model must be one of basic, logical, or random. Given {}".format(args.model))

	
	prefix, questions = load_data()

	score = 0
	num_invalid = 0		


	# Set up output file for data
	savedir = f"./{output_dir}/{model_name}/{path}"
	if not os.path.exists(savedir):
		os.makedirs(savedir)
	output_file = f"{savedir}/results.csv"
	
	with open(output_file, 'w', newline='') as output:
		writer = csv.writer(output)
		header = ["question_index", "response", "is_correct"]
		writer.writerow(header)

		# Loop through questions
		for i, q in enumerate(questions[:50]):
			query = q["input"]
			model_output = model(query, max_new_tokens=100)

			is_invalid = ans not in ["entailment", "non-entailment"]
			correct = "NaN" if is_invalid else q["target_scores"][ans]
			writer.writerow([i, ans, correct])
			score += 0 if is_invalid else q["target_scores"][ans]
			num_invalid += is_invalid
		
		writer.writerow(["accuracy", score / len(questions), "num_invalid", num_invalid])
	
	return score / len(questions), num_invalid



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
	parser.add_argument("--model", default="basic",
		help="The model to use. Options are basic, logical, or random")
	parser.add_argument("--path", default="meta-llama/Llama-2-13b-hf",
		help="The path to the LLM to use (huggingface path)")
	parser.add_argument("--output_dir", default="results",
		help="Directory to save results to")


	args = parser.parse_args()

#	print(args.model, args.output_dir, args.size)
	acc, inv = evaluate(args.model, args.path, args.output_dir)
	




