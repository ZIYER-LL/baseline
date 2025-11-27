from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 修改为你的实际模型路径
model_name = r"/work/2024/zhulei/intent-driven/qwen3-4b"  # 请修改为正确的模型路径

print(f"正在加载模型: {model_name}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✅ Tokenizer加载成功")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,  # 使用float16节省内存
        low_cpu_mem_usage=True
    )
    print("✅ 模型加载成功")
    print(f"模型设备: {model.device}")

except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("请检查:")
    print("1. 模型路径是否正确")
    print("2. 模型文件是否存在")
    print("3. 是否有足够的内存")
    raise

def run_qwen(prompt: str):
    print(f"\n正在处理prompt (长度: {len(prompt)})...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # 贪婪解码
            pad_token_id=tokenizer.eos_token_id  # 避免警告
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("✅ 生成完成")
        return result

    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return ""
