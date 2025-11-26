import json
from run_qwen import run_qwen              # 你的模型推理函数
from evaluation import evaluate_sample     # 自动评测模块
from rulebook import RULEBOOKS             # 你的 rulebook
import numpy as np


# ==========================
# 1. 载入测试集
# ==========================
print("加载测试集 test_dataset.json ...")
test_data = json.load(open("test_dataset.json", "r", encoding="utf-8"))

results = []

# 统计指标的容器
intent_type_list = []
service_type_list = []
joint_list = []

json_valid_list = []
missing_fields_list = []
extra_fields_list = []
qos_match_list = []
qos_valid_list = []


# ==========================
# 2. 对每条样本运行 baseline
# ==========================
for idx, item in enumerate(test_data):
    natural = item["natural_input"]
    intent_gt = item["intent_gt"]

    service = intent_gt["service_type"]
    rulebook = RULEBOOKS[service]

    print(f"\n========================")
    print(f"样本 {idx+1}/{len(test_data)}")
    print(f"自然语言输入：{natural}")

    result = evaluate_sample(
        natural_input=natural,
        intent_gt=intent_gt,
        rulebook=rulebook,
        model_fn=run_qwen    # Qwen-3-4B 推理函数
    )
    results.append(result)

    # Intent 指标
    i_type, s_type, joint = result["intent_ok"]
    intent_type_list.append(i_type)
    service_type_list.append(s_type)
    joint_list.append(joint)

    # Policy 指标
    json_valid_list.append(result["policy"]["json_valid"])
    missing_fields_list.append(not result["policy"]["missing_fields"])
    extra_fields_list.append(not result["policy"]["extra_fields"])
    qos_match_list.append(result["policy"]["qos_match"])
    qos_valid_list.append(result["policy"]["qos_valid"])


# ==========================
# 3. 输出汇总统计结果
# ==========================
def pct(x_list):
    return round(np.mean(x_list) * 100, 2)


print("\n\n========== Baseline 评测结果 ==========")
print(f"Intent 类型正确率: {pct(intent_type_list)}%")
print(f"Service 类型正确率: {pct(service_type_list)}%")
print(f"Intent 联合正确率: {pct(joint_list)}%")

print("\nPolicy 部分:")
print(f"JSON 合法率: {pct(json_valid_list)}%")
print(f"字段完整率（无缺失）: {pct(missing_fields_list)}%")
print(f"无多余字段率: {pct(extra_fields_list)}%")
print(f"QoS 完全匹配 rulebook: {pct(qos_match_list)}%")
print(f"QoS 合法值率（未瞎填）: {pct(qos_valid_list)}%")

# 端到端成功率（Intent联合正确 && JSON合法 && 字段完整 && QoS合法）
end2end = [
    joint_list[i]
    and json_valid_list[i]
    and missing_fields_list[i]
    and qos_valid_list[i]
    for i in range(len(test_data))
]

print(f"\n端到端成功率: {pct(end2end)}%")

# ==========================
# 4. 保存每条样本的详细结果
# ==========================
json.dump(
    results,
    open("baseline_results_detail.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=2
)

print("\n详细结果已保存到 baseline_results_detail.json")
print("汇总评测完成！\n")