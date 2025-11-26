RULEBOOKS = {
    "realtime_video": {
        "latency_max_ms": 80,
        "jitter_max_ms": 10,
        "packet_loss_rate_max": 0.01,
        "bandwidth_min_kbps": 2000,
        "reliability_min": 0.999,
        "priority": 3
    },
    "realtime_voice_call": {
        "latency_max_ms": 120,
        "jitter_max_ms": 20,
        "packet_loss_rate_max": 0.01,
        "bandwidth_min_kbps": 50,
        "reliability_min": 0.99,
        "priority": 2
    },
    "realtime_xr_gaming": {
        "latency_max_ms": 15,
        "jitter_max_ms": 5,
        "packet_loss_rate_max": 0.005,
        "bandwidth_min_kbps": 100000,
        "reliability_min": 0.999,
        "priority": 4
    },
    "streaming_video": {
        "latency_max_ms": 150,
        "jitter_max_ms": 30,
        "packet_loss_rate_max": 0.02,
        "bandwidth_min_kbps": 5000,
        "reliability_min": 0.995,
        "priority": 2
    },
    "streaming_live": {
        "latency_max_ms": 150,
        "jitter_max_ms": 15,
        "packet_loss_rate_max": 0.01,
        "bandwidth_min_kbps": 10000,
        "reliability_min": 0.99,
        "priority": 3
    },
    "file_transfer": {
        "latency_max_ms": 150,
        "jitter_max_ms": 50,
        "packet_loss_rate_max": 0.02,
        "bandwidth_min_kbps": 5000,
        "reliability_min": 0.999,
        "priority": 1
    },
    "iot_sensor": {
        "latency_max_ms": 50,
        "jitter_max_ms": 10,
        "packet_loss_rate_max": 0.001,
        "bandwidth_min_kbps": 200,
        "reliability_min": 0.9999,
        "priority": 3
    },
    "internet_access": {
        "latency_max_ms": 150,
        "jitter_max_ms": 30,
        "packet_loss_rate_max": 0.02,
        "bandwidth_min_kbps": 10000,
        "reliability_min": 0.98,
        "priority": 1
    },
    "urllc_control": {
        "latency_max_ms": 5,
        "jitter_max_ms": 1,
        "packet_loss_rate_max": 0.00001,
        "bandwidth_min_kbps": 1000,
        "reliability_min": 0.999999,
        "priority": 5
    }
}