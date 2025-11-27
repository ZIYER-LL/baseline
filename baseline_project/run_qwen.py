from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 修改为你的实际模型路径
model_name = r"/work/2024/zhulei/intent-driven/qwen3-4b"  # 请修改为正确的模型路径

print(f"正在加载模型: {model_name}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✅ Tokenizer加载成功")

    # 手动指定使用空闲的GPU（从nvidia-smi看GPU 1,2,3,6,7基本空闲）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 1},  # 指定使用GPU 1（基本空闲）
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

    # 检查输入
    print(f"输入token数量: {inputs['input_ids'].shape[1]}")

    try:
        # 使用最基本的参数避免警告
        generate_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "max_new_tokens": 512,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
        }

        # 只在需要时添加参数
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            generate_kwargs["eos_token_id"] = tokenizer.eos_token_id

        print("开始生成...")
        outputs = model.generate(**generate_kwargs)

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ 生成完成，输出长度: {len(result)}")
        return result

    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return ""
