from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# ------------------------------------------------
# 1️⃣ 加载 tokenizer
# tokenizer 负责：
# 文本 <-> token id 的转换
# ------------------------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# ------------------------------------------------
# 2️⃣ 加载模型
# GPT2LMHeadModel = GPT2 + language modeling head
#
# LM head 本质就是：
# hidden_state -> Linear -> vocab logits
# ------------------------------------------------
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 切换到推理模式
model.eval()



# ------------------------------------------------
# 3️⃣ 单步 forward
# 只计算一个 token 的生成
# ------------------------------------------------
def forward_one_step(model, input_ids, past_key_values):

    # torch.no_grad()
    # 表示关闭梯度计算
    # 推理时不需要反向传播
    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,

            # KV Cache
            # past_key_values 保存之前 token 的 attention key/value
            past_key_values=past_key_values,

            # use_cache=True
            # 告诉模型返回新的 KV cache
            use_cache=True
        )

    # outputs.logits shape:
    #
    # [batch, seq_len, vocab_size]
    #
    # 例如：
    # [1, 1, 50257]

    # 只取最后一个 token 的 logits
    logits = outputs.logits[:, -1, :]

    # ------------------------------------------------
    # 4️⃣ greedy decoding
    #
    # 从 vocab 中选 logits 最大的 token
    # ------------------------------------------------
    next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    # 返回：
    # 1️⃣ 生成的新 token
    # 2️⃣ 更新后的 KV cache
    return next_token, outputs.past_key_values



# ------------------------------------------------
# 5️⃣ 文本生成函数
# ------------------------------------------------
def generate(model, tokenizer, prompt, max_new_tokens=80):

    # ------------------------------------------------
    # tokenizer.encode
    #
    # 文本 -> token ids
    #
    # 例如：
    #
    # "Hello world"
    # ↓
    # [15496, 995]
    # ------------------------------------------------
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # KV cache 初始为空
    past = None


    # ------------------------------------------------
    # autoregressive generation
    #
    # 每次生成一个 token
    # ------------------------------------------------
    for _ in range(max_new_tokens):

        # 只输入最后一个 token
        # 因为之前 token 已经在 KV cache 中
        next_token, past = forward_one_step(
            model,
            input_ids[:, -1:],
            past
        )

        # 拼接新 token
        input_ids = torch.cat([input_ids, next_token], dim=-1)


    # ------------------------------------------------
    # tokenizer.decode
    #
    # token ids -> 文本
    # ------------------------------------------------
    return tokenizer.decode(input_ids[0])