"""Optional MQTT telemetry adapter; no broker connection is made unless explicitly called."""
import json

def publish_telemetry(payload, broker_host, topic='gridmind/telemetry', port=1883):
    try: import paho.mqtt.client as mqtt
    except ImportError as e: raise ImportError('Install paho-mqtt to publish telemetry') from e
    client=mqtt.Client(mqtt.CallbackAPIVersion.VERSION2); client.connect(broker_host,port,60)
    info=client.publish(topic,json.dumps(payload,default=float),qos=1); info.wait_for_publish(); client.disconnect()
