#!/usr/bin/env python3
"""
简单的模型测试脚本
"""

from run_qwen import run_qwen

def test_model():
    """测试模型是否能正常工作"""
    test_prompt = "你好，请介绍一下自己。"

    print("开始测试模型...")
    print(f"测试prompt: {test_prompt}")

    try:
        result = run_qwen(test_prompt)
        print("\n模型输出:")
        print("-" * 50)
        print(result)
        print("-" * 50)

        if result.strip():
            print("✅ 模型工作正常！")
        else:
            print("⚠️ 模型返回了空字符串")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_model()
