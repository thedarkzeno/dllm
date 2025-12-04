from dllm.pipelines.llada import LLaDAModelLM, LLaDAConfig
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("adalberto-temp/Qwen-Encoder")
vocab_size = len(tokenizer)

model_name = "GSAI-ML/LLaDA-8B-Base"
config = LLaDAConfig.from_pretrained(model_name)
config.embedding_size = vocab_size + 2
# config.hidden_size = 896
config.d_model = 896
config.eos_token_id = tokenizer.eos_token_id
config.pad_token_id = tokenizer.pad_token_id
config.mask_token_id = tokenizer.mask_token_id
config.vocab_size = vocab_size + 2
config.n_heads = 14
config.n_kv_heads = 14
config.n_layers = 24
config.mlp_ratio = 4
config.mlp_hidden_size = 4864
config.activation_type = "silu"
config.block_type = "sequential"
config.block_group_size = 1


print(config)

model = LLaDAModelLM(config)
print(model)
# count params
total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params}")

model.save_pretrained("./models/llada-small-portuguese")
tokenizer.save_pretrained("./models/llada-small-portuguese")