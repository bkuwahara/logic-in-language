import pandas as pd    
import random
import json

jsonObj = pd.read_json(path_or_buf="./datasets/snli_1.0/snli_1.0_dev.jsonl", lines=True)

entailments = jsonObj[jsonObj["gold_label"] == 'entailment']

non_factive = ["believes", "thinks", "assumes", "suspects"]
factive = ["knows", "sees", "learns", "understands", "recognizes", "remembers"]
names = ['John', 'James', 'Robert', 'Michael', 'David', 'Ava', 'Sophia', 'Olivia', 'William', 'Joseph', 'Isabella', 'Charlotte', 'Evelyn', 'Thomas', 'Amelia', 'Emma', 'Taylor', 'Charles', 'Richard', 'Abigail']

# Positive examples
# 200 each of pos_introspection (supported with K and B), neg_introspection (supported with B), consistency (B)





def generate_pos_int(name, statement):
    v1 = random.choice(factive+non_factive)
    v2 = random.choice(["believes", "knows", "understands", "recognizes", "thinks"])
    prem = f"{name} {v1} that {statement}"
    hyp = f"{name} {v2} that they {v1[:len(v1)-1]} that {statement}"
    return {"input" : f"Premise: {prem} Hypothesis: {hyp}", "target_scores" : {"entailment": 1, "non-entailment": 0}}


def generate_neg_int(name, statement):
    v1 = random.choice(non_factive)
    v1 = v1[:len(v1)-1]
    v2 = random.choice(["believes", "knows", "understands", "recognizes", "thinks"])
    prem = f"{name} does not {v1} that {statement}"
    hyp = f"{name} {v2} that they do not {v1} that {statement}"
    # Need one name, one statement, one verb
    return {"input" : f"Premise: {prem} Hypothesis: {hyp}", "target_scores" : {"entailment": 1, "non-entailment": 0}}

def generate_consistency(name, statement):
    v1 = random.choice(non_factive)
    v2 = random.choice(non_factive)
    v2 = v2[:len(v2)-1]
    prem = f"{name} {v1} that {statement}."
    hyp = f"{name} does not {v1} that it is not that case that {statement}"
    return {"input": f"Premise: {prem} Hypothesis: {hyp}", "target_scores" : {"entailment": 1, "non-entailment": 0}}


# K/B(a,x) -> B(a,!x)
def generate_contradiction(name, statement):
    v1 = random.choice(factive+non_factive)
    v2 = random.choice(factive+non_factive)
    prem = f"{name} {v1} that {statement}"
    hyp = f"{name} {v2} that it is not the case that {statement}"
    return {"input" : f"Premise: {prem} Hypothesis: {hyp}", "target_scores" : {"entailment": 0, "non-entailment": 1}}

# B(a,x) -> B(a,K(a,x))
def generate_overconfidence(name, statement):
    v1 = random.choice(non_factive)
    v2 = random.choice(factive)
    prem = f"{name} {v1} that {statement}"
    hyp = f"{name} {v1} that they {v2} that {statement}"
    return {"input" : f"Premise: {prem} Hypothesis: {hyp}", "target_scores" : {"entailment": 0, "non-entailment": 1}}


def generate_dataset(template_counts):
    generators = [generate_pos_int, generate_neg_int, generate_consistency, generate_contradiction, generate_overconfidence]
    n_total = sum(template_counts)
    statements = entailments.sample(n=n_total)
    questions = []
    i = 0 # Tracks index in statements

    for (j, row) in statements.iterrows():
        statement = row["sentence1"]
        name = random.choice(names)
        question = generators[i](name, statement)
        questions.append(question)

        template_counts[i] -= 1
        if not template_counts[i]:
            i += 1

    dataset = {"name": "kd45 reasoning", "description": "Decide if one statement entails the next", "examples" : questions}
    js = json.dumps(dataset)
    with open("./datasets/kd45_reasoning.json", 'w') as outfile:
        outfile.write(js)



def generate_mixture(n_from_base, n_from_kd45):
    with open("./datasets/task.json") as f:
        base = json.load(f)
    output = {"name" : "mixed epistemic reasoning", 
              "description" : "A mixture of the BIG-bench epistemic reasoning dataset and kd45 reasoning dataset"}
    base_questions = random.choices(base["examples"], k=n_from_base)

    with open("./datasets/kd45_reasoning.json") as f:
        kd45 = json.load(f)
    
    kd45_questions = random.choices(kd45["examples"], k=n_from_kd45)
    output["examples"] = base_questions+kd45_questions
    js = json.dumps(output)
    with open("./datasets/mixed_reasoning.json", 'w') as outfile:
        outfile.write(js)
    


if __name__ == '__main__':
    generate_dataset([100] * 5)
    generate_mixture(250, 250)