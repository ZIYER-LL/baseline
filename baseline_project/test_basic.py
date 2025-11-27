#!/usr/bin/env python3
"""
基础环境测试
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定GPU 1

import torch
from transformers import AutoTokenizer

print("=== 基础环境测试 ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} - {props.total_memory // 1024**3}GB")

# 测试tokenizer
model_name = r"/work/2024/zhulei/intent-driven/qwen3-4b"

try:
    print(f"\n测试tokenizer加载: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✅ Tokenizer加载成功")

    # 测试编码解码
    test_text = "你好，世界！"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)

    print(f"原文: {test_text}")
    print(f"Token数量: {len(tokens)}")
    print(f"解码: {decoded}")

    print("✅ 基础功能正常")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
