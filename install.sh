#!/bin/bash

# Move to the project directory
cd ~/Documents/trafficLights

# Update and install dependencies automatically with -y
sudo apt update
sudo apt install -y curl mosquitto-clients
sudo apt-get install -y gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-plugins-good gstreamer1.0-libav gstreamer1.0-tools

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Manually add uv to the script's PATH so the next commands work immediately
export PATH="$HOME/.local/bin:$PATH"

# Setup Python environment
uv sync
source .venv/bin/activate 
uv pip install -r requirements.txt

# Configure Network (added '|| true' so the script doesn't crash if the connection doesn't exist yet)
sudo nmcli connection delete enx00e04c36131a || true
sudo nmcli connection add type ethernet con-name "CameraNet" ifname enx00e04c36131a ipv4.method manual ipv4.addresses 192.168.1.200/24
sudo nmcli connection up "CameraNet"
ip addr show enx00e04c36131a

# Define the aliases
RUNAPP_ALIAS="alias runapp='cd ~/Documents/trafficLights && source .venv/bin/activate && uv run gui/main.py'"
VRED_ALIAS="alias vred='mosquitto_pub -h test.mosquitto.org -t ub-traffic-light/signals/vertical -m \"red\"'"

# Append them to .bashrc if they don't already exist
grep -qF "$RUNAPP_ALIAS" ~/.bashrc || echo "$RUNAPP_ALIAS" >> ~/.bashrc
grep -qF "$VRED_ALIAS" ~/.bashrc || echo "$VRED_ALIAS" >> ~/.bashrc

# Run the app directly (aliases don't expand in scripts)
echo "Installation complete! Starting the application..."
uv run gui/main.py
