from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "/work/2024/zhulei/intent-driven/qwen3-4b"   # 你自己的路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)

def run_qwen(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,          # baseline 用低温生成更加稳定
        do_sample=False           # baseline 先关闭随机采样
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)