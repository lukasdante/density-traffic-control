import paho.mqtt.client as mqtt
from typing import Literal

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "ub-traffic-light/signals/"
MESSAGE = "red"

def publish(lane: Literal["vertical", "horizontal"]):
    """Publishes a Python dictionary as JSON to the MQTT topic."""

    client = mqtt.Client()
    topic = f"{TOPIC}{lane}"
    client.connect(BROKER, PORT)
    client.publish(topic, MESSAGE)
    client.disconnect()
    print(f"Published: {MESSAGE} to {lane} traffic light.")

if __name__ == "__main__":
    publish("horizontal")


# Command
# mosquitto_pub -h test.mosquitto.org -t "ub-traffic-light/signals/vertical" -m "red"
# mosquitto_pub -h test.mosquitto.org -t "ub-traffic-light/signals/horizontal" -m "red"