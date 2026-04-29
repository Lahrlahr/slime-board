from transformers import AutoTokenizer, AutoModel

model_path = "/data/huangguang/model/Qwen/Qwen3-VL-2B-Instruct/"


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

new_specials = [f"<|action_{i}|>" for i in range(1024)]

tokenizer.add_special_tokens({
    "additional_special_tokens": new_specials
})

texts = [
    "hello world",
    "do <|action_0|> now",
    "<|action_1|> then <|action_2|>",
    "mix action_0 without special format",
]

for text in texts:
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"\nInput: {text}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {ids}")

model = AutoModel.from_pretrained(model_path)

model.resize_token_embeddings(len(tokenizer))

# 3. 保存
save_path = "/data/huangguang/model/Qwen/Qwen3-VL-2B-Instruct-expand"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)