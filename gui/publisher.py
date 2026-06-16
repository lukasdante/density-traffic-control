import argparse
import uuid
from typing import Literal

import paho.mqtt.client as mqtt


BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC_PREFIX = "ub-traffic-light/signals"
MESSAGE = "red"


def publish(
    lane: Literal["vertical", "horizontal"],
    message: str = MESSAGE,
    broker: str = BROKER,
    port: int = PORT,
) -> None:
    """Publish a traffic-light command to the lane-specific MQTT topic."""
    topic = f"{TOPIC_PREFIX}/{lane}"
    client_id = f"tl-publisher-{uuid.uuid4().hex[:8]}"
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
    )

    client.connect(broker, port, keepalive=30)
    result = client.publish(topic, message, qos=0, retain=False)
    result.wait_for_publish()
    client.disconnect()
    print(f'Published "{message}" to {topic} via {broker}:{port}')


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish MQTT traffic-light commands.")
    parser.add_argument(
        "lane",
        choices=("vertical", "horizontal"),
        help="Traffic-light lane topic suffix.",
    )
    parser.add_argument(
        "message",
        nargs="?",
        default=MESSAGE,
        help='Message to publish. Defaults to "red".',
    )
    parser.add_argument("--broker", default=BROKER, help="MQTT broker host.")
    parser.add_argument("--port", default=PORT, type=int, help="MQTT broker port.")
    args = parser.parse_args()

    publish(args.lane, args.message, args.broker, args.port)


if __name__ == "__main__":
    main()


# Command
# python gui/publisher.py vertical red
# python gui/publisher.py horizontal red
# mosquitto_pub -h broker.hivemq.com -t "ub-traffic-light/signals/vertical" -m "red"
# mosquitto_pub -h broker.hivemq.com -t "ub-traffic-light/signals/horizontal" -m "red"
