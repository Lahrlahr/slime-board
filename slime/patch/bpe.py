from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import json
# ========= 1. 构造示例数据 =========
# 假设你的数据是 >=0 的整数序列
int_sequences = [
    [2,5,4,6],
    # [221, 352, 253, 319, 329,221, 352, 253, 319, 329,221, 352, 253, 319, 329],
    # [221, 232, 248, 258, 248,221, 352, 101, 202, 123, 324,101, 202, 123, 324,253, 319, 329,357, 317, 327, 421, 122,357, 317, 327, 421, 122],
    # [357, 317, 327, 421, 122],
]

# ========= 2. 转成字符串 =========
# ⚠️ 推荐：限制到 0–255（ByteLevel最稳）
def ints_to_str(seq):
    return "".join(chr(x) for x in seq)

text_data = "".join(ints_to_str(seq) for seq in int_sequences)

# 写入文件
with open("train.txt", "w", encoding="utf-8") as f:
    f.write(text_data)

# ========= 3. 初始化 tokenizer =========
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 不乱切，按 byte 处理
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

# ========= 4. trainer =========
trainer = BpeTrainer(
    vocab_size=1024,
    min_frequency=1,
)

# ========= 5. 训练 =========
tokenizer.train(["train.txt"], trainer)

model = tokenizer.model


tokenizer.save("bpe_1024.json")
with open("bpe_1024.json", "r", encoding="utf-8") as f:
    data = json.load(f)

merges = data["model"]["merges"]

print(merges[:20])
# ========= 7. 测试 =========
test_seq = [1, 2, 3, 4, 5]
test_str = ints_to_str(test_seq)

output = tokenizer.encode(test_str)

print("原始序列:", test_seq)
print("编码 tokens:", output.tokens)
print("编码 ids:", output.ids)