from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
# tokenizer = AutoTokenizer.from_pretrained("/data/huangguang/model/Qwen/Qwen2.5-1.5B-Instruct-expand")

model_path = "/data/huangguang/model/Qwen/Qwen2.5-1.5B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto",
    device_map="auto")


model.resize_token_embeddings(len(tokenizer))

new_specials = [f"<|action_{i}|>" for i in range(6000)]
tokenizer.add_special_tokens({
    "additional_special_tokens": new_specials
})

model.resize_token_embeddings(len(tokenizer))

save_path = "/data/huangguang/model/Qwen/Qwen2.5-1.5B-expand"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

hidden_size = model.config.hidden_size
model.lm_head = nn.Linear(hidden_size, 1, bias=False)
model.config.tie_word_embeddings = False
nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
save_path = "/data/huangguang/model/Qwen/Qwen2.5-1.5B-critic"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)