import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
torch_dtype=torch.float16
llama_path = "/w/339/bkuwahara/llama_model/13b"

print(torch.cuda.is_available())

#tokenizer = LlamaTokenizer.from_pretrained(llama_path, device_map="auto", offload_folder="offload", offload_state_dict=True)
#model = LlamaForCausalLM.from_pretrained(llama_path, device_map="auto",offload_folder="offload",offload_state_dict=True)
tokenizer = LlamaTokenizer.from_pretrained(llama_path)
model = LlamaForCausalLM.from_pretrained(llama_path, device_map="auto", offload_folder="offload", torch_dtype=torch.float16)

# tokenizer = LlamaTokenizer.from_pretrained(llama_path)
# model = LlamaForCausalLM.from_pretrained(llama_path)
device="cuda"
model.to(device)




