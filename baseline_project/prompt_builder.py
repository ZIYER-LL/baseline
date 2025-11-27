import json

def build_prompt_intent(natural_input: str):
    return f"""从以下用户输入中识别意图类型和业务类型，直接输出JSON格式：

intent_type选项：slice_create, slice_qos_modify, route_preference, access_control
service_type选项：realtime_video, realtime_voice_call, realtime_xr_gaming, streaming_video, streaming_live, file_transfer, iot_sensor, internet_access, urllc_control

用户输入：{natural_input}

输出格式：{{"intent_type": "xxx", "service_type": "yyy"}}"""

def build_prompt_policy(intent_pred: dict, rulebook: dict):
    # 根据intent_type生成合适的policy_id和routing_pref
    intent_type = intent_pred.get("intent_type", "slice_create")
    service_type = intent_pred.get("service_type", "realtime_video")

    # 生成policy_id
    policy_id = f"{service_type}_{intent_type}"

    # 根据intent_type确定routing_pref
    routing_prefs = {
        "slice_create": "low_latency",
        "slice_qos_modify": "qos_optimized",
        "route_preference": "priority_routing",
        "access_control": "secure_routing"
    }
    routing_pref = routing_prefs.get(intent_type, "default")

    # allowed_ue_group可以根据service_type确定
    ue_groups = {
        "realtime_video": "video_users",
        "realtime_voice_call": "voice_users",
        "realtime_xr_gaming": "gaming_users",
        "streaming_video": "streaming_users",
        "streaming_live": "live_users",
        "file_transfer": "data_users",
        "iot_sensor": "iot_devices",
        "internet_access": "internet_users",
        "urllc_control": "urllc_devices"
    }
    allowed_ue_group = ue_groups.get(service_type, "default_group")

    return f"""根据以下信息生成网络策略JSON：

Intent: {json.dumps(intent_pred, ensure_ascii=False)}
QoS规则: {json.dumps(rulebook, ensure_ascii=False)}

要求：直接输出JSON，不要任何解释或额外文字。

{{
  "policy_id": "{policy_id}",
  "qos_profile": {json.dumps(rulebook, ensure_ascii=False)},
  "routing_pref": "{routing_pref}",
  "allowed_ue_group": "{allowed_ue_group}"
}}"""
