import json

# ================================
# 工具函数：安全解析 JSON
# ================================
def safe_json_parse(text: str):
    """
    将模型输出尽可能解析成 JSON。
    baseline 模型经常输出多余文字，此函数尽量从中提取出有效 JSON。
    返回 (json_object, True) 或 (None, False)
    """
    # 1. 第一种：直接解析
    try:
        return json.loads(text), True
    except:
        pass

    # 2. 第二种：尝试截取文本中第一对 { ... }
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            return json.loads(json_str), True
    except:
        pass

    # 全部失败
    return None, False


# ================================
# Intent 评测
# ================================
def evaluate_intent(intent_pred: dict, intent_gt: dict):
    """
    intent_pred: 模型预测的 intent JSON
    intent_gt:   测试集中标注的意图
    返回：
      - intent_type 是否正确 (True/False)
      - service_type 是否正确 (True/False)
      - 两者是否完全正确 (True/False)
    """
    intent_type_correct = (intent_pred.get("intent_type") == intent_gt["intent_type"])
    service_type_correct = (intent_pred.get("service_type") == intent_gt["service_type"])
    joint_correct = intent_type_correct and service_type_correct

    return intent_type_correct, service_type_correct, joint_correct


# ================================
# QoS 合法性检查（避免瞎填）
# ================================
def check_qos_range(k: str, v: float):
    """
    为避免“瞎填”，定义一个宽松但合理的范围。
    你可以根据真实需求进一步加强范围。
    """
    ranges = {
        "latency_max_ms": (0, 500),
        "jitter_max_ms": (0, 200),
        "packet_loss_rate_max": (0.0, 0.2),
        "bandwidth_min_kbps": (1, 10_000_000),
        "reliability_min": (0.9, 1.0),
        "priority": (1, 5)
    }

    mn, mx = ranges[k]
    return mn <= v <= mx


# ================================
# Policy 评测
# ================================
def evaluate_policy(policy_pred: dict, rulebook: dict):
    """
    对模型生成的 Policy JSON 做结构、字段、QoS 检查。
    返回指标字典：
      - json_valid        JSON 是否可解析
      - missing_fields    是否有字段缺失
      - extra_fields      是否多加字段
      - qos_match         QoS 是否完全等于 rulebook
      - qos_valid         QoS 是否落在合理范围内
    """

    required_top_keys = ["policy_id", "qos_profile", "routing_pref", "allowed_ue_group"]

    # 1. 顶层字段检查
    for k in required_top_keys:
        if k not in policy_pred:
            return {
                "json_valid": True,
                "missing_fields": True,
                "extra_fields": False,
                "qos_match": False,
                "qos_valid": False
            }

    qos_pred = policy_pred["qos_profile"]

    # 2. 字段缺失检查
    missing_fields = any(k not in qos_pred for k in rulebook)

    # 3. 多余字段检查
    extra_fields = any(k not in rulebook for k in qos_pred)

    # 4. QoS 与 rulebook 精确匹配
    qos_match = all(
        (k in qos_pred) and (qos_pred[k] == v)
        for k, v in rulebook.items()
    )

    # 5. QoS 数值合法性检查
    qos_valid = all(
        (k in qos_pred) and check_qos_range(k, qos_pred[k])
        for k in rulebook.keys()
    )

    return {
        "json_valid": True,
        "missing_fields": missing_fields,
        "extra_fields": extra_fields,
        "qos_match": qos_match,
        "qos_valid": qos_valid
    }


# ================================
# 智能JSON解析（改进版）
# ================================
def smart_json_parse(text: str, json_type="intent"):
    """
    改进的JSON解析函数，能够从模型输出中智能提取正确的JSON
    """
    import re

    # 1. 先尝试标准解析
    result, success = safe_json_parse(text)
    if success:
        return result, True

    # 2. 智能提取逻辑
    print(f"标准解析失败，尝试智能提取{json_type} JSON...")

    # 查找所有JSON对象
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    json_matches = re.findall(json_pattern, text)

    print(f"  找到 {len(json_matches)} 个JSON对象")

    best_json = None
    for i, json_str in enumerate(json_matches):
        try:
            parsed = json.loads(json_str)

            if json_type == "intent":
                # Intent JSON验证
                intent_type = parsed.get('intent_type', '')
                service_type = parsed.get('service_type', '')
                if (intent_type and intent_type not in ['xxx', 'intent_type'] and
                    service_type and service_type not in ['yyy', 'service_type']):
                    best_json = parsed
                    print(f"  ✅ 找到有效的Intent JSON: {intent_type}, {service_type}")
                    break

            elif json_type == "policy":
                # Policy JSON验证
                required_fields = ["policy_id", "qos_profile", "routing_pref", "allowed_ue_group"]
                if all(field in parsed for field in required_fields):
                    qos_profile = parsed.get("qos_profile", {})
                    if isinstance(qos_profile, dict) and len(qos_profile) > 0:
                        best_json = parsed
                        print(f"  ✅ 找到有效的Policy JSON: policy_id={parsed.get('policy_id')}")
                        break

        except Exception as e:
            continue

    if best_json:
        return best_json, True
    else:
        print(f"  ❌ 无法找到有效的{json_type} JSON")
        return None, False

# ================================
# 端到端评测（自然语言 → 模型 → Intent + Policy）
# ================================
def evaluate_sample(natural_input: str, intent_gt: dict, rulebook: dict, model_fn):
    """
    natural_input: 原始自然语言
    intent_gt:     测试集标注的意图（intent_type / service_type）
    rulebook:      该业务类型对应的默认 QoS
    model_fn:      一个函数，用于模型推理 (prompt → text)

    返回一个字典，包含：
      - intent_ok: (intent_type_correct, service_type_correct, joint_correct)
      - policy:    policy 各指标
    """

    # ==========================
    # Step 1：构造 Intent prompt
    # ==========================
    intent_prompt = f"""任务：从用户输入中识别intent_type和service_type。

可选值：
intent_type: slice_create, slice_qos_modify, route_preference, access_control
service_type: realtime_video, realtime_voice_call, realtime_xr_gaming, streaming_video, streaming_live, file_transfer, iot_sensor, internet_access, urllc_control

用户输入：{natural_input}

只输出JSON，不要任何其他文字："""

    # 模型输出 Intent
    intent_raw = model_fn(intent_prompt)
    intent_pred, ok_intent = smart_json_parse(intent_raw, "intent")

    if not ok_intent:
        # Intent 都无法解析 → policy 也不测试了
        return {
            "intent_ok": (False, False, False),
            "policy": {
                "json_valid": False,
                "missing_fields": True,
                "extra_fields": False,
                "qos_match": False,
                "qos_valid": False
            }
        }

    # Intent 评测
    intent_scores = evaluate_intent(intent_pred, intent_gt)


    # ==========================
    # Step 2：构造 Policy prompt
    # ==========================
    policy_prompt = f"""生成网络策略JSON，只输出JSON格式：

{{
  "policy_id": "{intent_pred.get('service_type', 'unknown')}_{intent_pred.get('intent_type', 'unknown')}",
  "qos_profile": {json.dumps(rulebook, ensure_ascii=False)},
  "routing_pref": "low_latency",
  "allowed_ue_group": "{intent_pred.get('service_type', 'unknown')}_users"
}}

只输出以上JSON，不要任何解释文字："""

    # 模型输出 Policy
    policy_raw = model_fn(policy_prompt)
    policy_pred, ok_policy = smart_json_parse(policy_raw, "policy")

    if not ok_policy:
        return {
            "intent_ok": intent_scores,
            "policy": {
                "json_valid": False,
                "missing_fields": True,
                "extra_fields": False,
                "qos_match": False,
                "qos_valid": False
            }
        }

    # Policy 评测
    policy_scores = evaluate_policy(policy_pred, rulebook)

    return {
        "intent_ok": intent_scores,
        "policy": policy_scores
    }
