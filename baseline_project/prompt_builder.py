import json

def build_prompt_intent(natural_input: str):
    return f"""任务：从用户输入中识别intent_type和service_type。

可选值：
intent_type: slice_create, slice_qos_modify, route_preference, access_control
service_type: realtime_video, realtime_voice_call, realtime_xr_gaming, streaming_video, streaming_live, file_transfer, iot_sensor, internet_access, urllc_control

用户输入：{natural_input}

只输出JSON，不要任何其他文字："""

def build_prompt_policy(intent_pred: dict, rulebook: dict):
    return f"""生成网络策略JSON，只输出JSON格式：

{{
  "policy_id": "{intent_pred.get('service_type', 'unknown')}_{intent_pred.get('intent_type', 'unknown')}",
  "qos_profile": {json.dumps(rulebook, ensure_ascii=False)},
  "routing_pref": "low_latency",
  "allowed_ue_group": "{intent_pred.get('service_type', 'unknown')}_users"
}}

只输出以上JSON，不要任何解释文字："""

