import json
import os
import random
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
import csv
import argparse
os.chdir("/w/339/bkuwahara/csc2542")

from models import LlamaBasic, LlamaLogical, RandModel

def load_data():
	with open("./datasets/task.json") as data_file:
		data = json.load(data_file)

	prefix = data["task_prefix"]
	questions = data["examples"]
	return prefix, questions




def evaluate(model_name, size, output_dir):

	# Load in the model
	if model_name == "llama_basic":
		model = LlamaBasic(size)
	elif model_name == "llama_logical":
		model = LlamaLogical(size)
	elif model_name == "random":
		model = RandModel(size)
	else:
		raise ValueError("Model must be one of llama_basic, llama_logical, or random. Given {}".format(args.model))

	
	prefix, questions = load_data()

	score = 0
	num_invalid = 0		


	# Set up output file for data
	savedir = f"./{output_dir}/{model_name}"
	if not os.path.exists(savedir):
		os.makedirs(savedir)
	output_file = f"{savedir}/{size}-results.csv"
	
	with open(output_file, 'w', newline='') as output:
		writer = csv.writer(output)
		header = ["question_index", "response", "is_correct"]
		writer.writerow(header)

		# Loop through questions
		for i, q in tqdm(enumerate(questions)):
			query = q["input"]
			prompt = prefix + query + "\nYour answer: "
			model_output = model(prompt)
			ans = model_output.split("Your answer: ")[1].split()[0]
			

			is_invalid = ans not in ["entailment", "non-entailment"]
			correct = "NaN" if is_invalid else q["target_scores"][ans]
			writer.writerow([i, ans, correct])
			score += 0 if is_invalid else q["target_scores"][ans]
			num_invalid += is_invalid
		
		writer.writerow(["accuracy", score / len(questions), "num_invalid", num_invalid])
	
	return score / len(questions), num_invalid



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
	parser.add_argument("--model", default="llama_basic",
		help="The model to use. Options are llama_basic, llama_logical, or random")
	parser.add_argument("--size", default="13B",
		help="Size of the model to use. Options are 7B or 13B")
	parser.add_argument("--output_dir", default="results",
		help="Directory to save results to")


	args = parser.parse_args()

#	print(args.model, args.output_dir, args.size)
	acc, inv = evaluate(args.model, args.size, args.output_dir)
	




