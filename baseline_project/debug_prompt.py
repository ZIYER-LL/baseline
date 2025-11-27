import json
from run_qwen import run_qwen
from evaluation import safe_json_parse
from rulebook import RULEBOOKS
from prompt_builder import build_prompt_intent, build_prompt_policy


def debug_single_sample(sample_idx: int, test_data: list):
    """
    调试单条样本的完整推理过程
    """
    if sample_idx < 0 or sample_idx >= len(test_data):
        print(f"错误：样本索引 {sample_idx} 超出范围 [0, {len(test_data)-1}]")
        return

    item = test_data[sample_idx]
    natural = item["natural_input"]
    intent_gt = item["intent_gt"]

    service = intent_gt["service_type"]
    rulebook = RULEBOOKS[service]

    print(f"\n{'='*80}")
    print(f"调试样本 {sample_idx + 1}/{len(test_data)}")
    print(f"{'='*80}")
    print(f"自然语言输入: {natural}")
    print(f"标注意图: {json.dumps(intent_gt, ensure_ascii=False, indent=2)}")
    print(f"业务类型: {service}")
    print(f"对应rulebook: {json.dumps(rulebook, ensure_ascii=False, indent=2)}")

    # ================================
    # Step 1: Intent 推理
    # ================================
    print(f"\n{'-'*60}")
    print("Step 1: Intent 识别")
    print(f"{'-'*60}")

    # 构建intent prompt
    intent_prompt = build_prompt_intent(natural)
    print("Intent Prompt:")
    print("-" * 40)
    print(intent_prompt.strip())
    print("-" * 40)

    # 模型原始输出
    intent_raw = run_qwen(intent_prompt)
    print("\n模型原始输出 (Intent):")
    print("-" * 40)
    print(repr(intent_raw))  # 用repr显示原始字符串，包括特殊字符
    print("-" * 40)

    # 解析结果
    intent_pred, ok_intent = safe_json_parse(intent_raw)
    print(f"\n解析成功: {ok_intent}")
    if ok_intent:
        print("解析后的JSON:")
        print(json.dumps(intent_pred, ensure_ascii=False, indent=2))
    else:
        print("❌ Intent解析失败，无法进行后续步骤")
        return

    # ================================
    # Step 2: Policy 推理
    # ================================
    print(f"\n{'-'*60}")
    print("Step 2: Policy 生成")
    print(f"{'-'*60}")

    # 构建policy prompt
    policy_prompt = build_prompt_policy(intent_pred, rulebook)
    print("Policy Prompt:")
    print("-" * 40)
    print(policy_prompt.strip())
    print("-" * 40)

    # 模型原始输出
    policy_raw = run_qwen(policy_prompt)
    print("\n模型原始输出 (Policy):")
    print("-" * 40)
    print(repr(policy_raw))  # 用repr显示原始字符串
    print("-" * 40)

    # 解析结果
    policy_pred, ok_policy = safe_json_parse(policy_raw)
    print(f"\n解析成功: {ok_policy}")
    if ok_policy:
        print("解析后的JSON:")
        print(json.dumps(policy_pred, ensure_ascii=False, indent=2))
    else:
        print("❌ Policy解析失败")


def debug_interactive(test_data: list):
    """
    交互式调试界面
    """
    while True:
        print(f"\n{'='*60}")
        print("Prompt调试工具")
        print(f"{'='*60}")
        print(f"总共 {len(test_data)} 条测试样本")
        print("输入样本编号 (1-{len(test_data)}) 进行调试")
        print("输入 'all' 查看所有样本摘要")
        print("输入 'q' 退出")

        choice = input("\n请选择: ").strip().lower()

        if choice == 'q':
            break
        elif choice == 'all':
            print(f"\n{'='*80}")
            print("所有测试样本摘要")
            print(f"{'='*80}")
            for i, item in enumerate(test_data):
                natural = item["natural_input"]
                intent_gt = item["intent_gt"]
                print(f"{i+1:2d}. {natural[:50]}{'...' if len(natural) > 50 else ''}")
                print(f"    意图: {intent_gt['intent_type']} / {intent_gt['service_type']}")
        else:
            try:
                idx = int(choice) - 1
                debug_single_sample(idx, test_data)
            except ValueError:
                print("❌ 请输入有效的数字或 'all' 或 'q'")


def debug_batch_summary(test_data: list, max_samples=None):
    """
    批量调试，显示所有样本的简要结果
    """
    if max_samples is None:
        max_samples = len(test_data)

    print(f"\n{'='*100}")
    print(f"批量调试摘要 (前 {max_samples} 条样本)")
    print(f"{'='*100}")
    print("样本 | 解析状态 | 输入文本")
    print("-" * 100)

    for idx in range(min(max_samples, len(test_data))):
        item = test_data[idx]
        natural = item["natural_input"]
        intent_gt = item["intent_gt"]
        service = intent_gt["service_type"]
        rulebook = RULEBOOKS[service]

        # Intent推理
        intent_prompt = build_prompt_intent(natural)
        intent_raw = run_qwen(intent_prompt)
        intent_pred, ok_intent = safe_json_parse(intent_raw)

        # Policy推理
        policy_raw = ""
        ok_policy = False
        if ok_intent:
            policy_prompt = build_prompt_policy(intent_pred, rulebook)
            policy_raw = run_qwen(policy_prompt)
            policy_pred, ok_policy = safe_json_parse(policy_raw)

        # 输出结果
        status = f"Intent: {'✅' if ok_intent else '❌'} | Policy: {'✅' if ok_policy else '❌'}"
        print("<3")
        print(status)
        print(f"输入: {natural}")
        print("-" * 100)


def main():
    """
    主函数
    """
    print("加载测试数据集...")
    try:
        test_data = json.load(open("test_dataset.json", "r", encoding="utf-8"))
        print(f"✅ 成功加载 {len(test_data)} 条测试样本")
    except FileNotFoundError:
        print("❌ 找不到 test_dataset.json 文件")
        return
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        return

    print("\n选择调试模式:")
    print("1. 交互式调试 (逐条查看详细过程)")
    print("2. 批量摘要 (快速查看所有样本结果)")
    print("3. 调试单条样本 (指定索引)")

    choice = input("\n请选择模式 (1/2/3): ").strip()

    if choice == '1':
        debug_interactive(test_data)
    elif choice == '2':
        max_samples = input("输入要查看的最大样本数 (默认全部): ").strip()
        max_samples = int(max_samples) if max_samples.isdigit() else None
        debug_batch_summary(test_data, max_samples)
    elif choice == '3':
        idx_input = input("输入样本索引 (1-{len(test_data)}): ").strip()
        try:
            idx = int(idx_input) - 1
            debug_single_sample(idx, test_data)
        except ValueError:
            print("❌ 请输入有效的数字")
    else:
        print("❌ 无效选择")


if __name__ == "__main__":
    main()
