#!/usr/bin/env python3
"""
测试指定GPU的脚本
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 指定使用GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

print("=== 测试指定GPU ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} - {props.total_memory // 1024**3}GB")

# 模型路径
model_name = r"/work/2024/zhulei/intent-driven/qwen3-4b"

try:
    print(f"\n加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # 现在只会看到GPU 1
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    print(f"✅ 模型加载成功，设备: {model.device}")

    # 测试推理
    test_prompt = "你好"
    print(f"\n测试推理: {test_prompt}")

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✅ 推理成功: {result[:100]}...")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
