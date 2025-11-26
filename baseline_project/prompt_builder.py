import json

def build_prompt_intent(natural_input: str):
    return f"""
你是一个6G核心网的意图识别助手。

任务：
从用户的自然语言中识别两个字段：
1）intent_type（取值范围：slice_create / slice_qos_modify / route_preference / access_control）
2）service_type（取值范围：realtime_video / realtime_voice_call / realtime_xr_gaming /
                 streaming_video / streaming_live / file_transfer / iot_sensor /
                 internet_access / urllc_control）

要求：
- 严格输出JSON
- 只包含 intent_type 和 service_type
- 不要输出任何解释

用户输入：{natural_input}
"""

def build_prompt_policy(intent_pred: dict, rulebook: dict):
    return f"""
你是一个6G核心网策略生成助手。

任务：
根据意图（intent）和规则库（rulebook），生成策略 JSON（policy）。

要求：
- 严格输出 JSON
- 包含 policy_id, qos_profile, routing_pref, allowed_ue_group
- qos_profile 必须严格使用 rulebook 中的数值
- 不要添加多余字段
- 不要生成解释

Intent:
{json.dumps(intent_pred, ensure_ascii=False)}

Rulebook:
{json.dumps(rulebook, ensure_ascii=False)}

请输出策略 JSON：
"""