import paho.mqtt.client as mqtt
from typing import Literal

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "ub-traffic-light/signals/"
MESSAGE = "red"

import paho.mqtt.publish as publish_helper

def publish(lane: Literal["vertical", "horizontal"]):
    topic = f"{TOPIC}{lane}"
    
    # This one line replaces connect, publish, wait, and disconnect
    publish_helper.single(topic, payload=MESSAGE, hostname=BROKER)
    
    print(f"Published: {MESSAGE} to {lane} traffic light.")

if __name__ == "__main__":
    publish("horizontal")


# Command
# mosquitto_pub -h broker.hivemq.com -t "ub-traffic-light/signals/vertical" -m "red"
# mosquitto_pub -h broker.hivemq.com -t "ub-traffic-light/signals/horizontal" -m "red"