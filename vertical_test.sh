#!/usr/bin/env bash
set -euo pipefail

mosquitto_pub -h broker.hivemq.com -p 1883 -t "ub-traffic-light/signals/vertical" -m "red"
