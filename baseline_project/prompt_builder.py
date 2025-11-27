import json

def build_prompt_intent(natural_input: str):
    return f"""从以下用户输入中识别意图类型和业务类型，直接输出JSON格式：

intent_type选项：slice_create, slice_qos_modify, route_preference, access_control
service_type选项：realtime_video, realtime_voice_call, realtime_xr_gaming, streaming_video, streaming_live, file_transfer, iot_sensor, internet_access, urllc_control

用户输入：{natural_input}

输出格式：{{"intent_type": "xxx", "service_type": "yyy"}}"""

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
