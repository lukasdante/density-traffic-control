#include <WiFi.h>
#include <esp_now.h>
#include "esp_wifi.h"
#include <PubSubClient.h>

// ---------- CONFIG ----------
const char* WIFI_SSID   = "ub-hs-2.4G";      // 2.4 GHz Wi-Fi SSID
const char* WIFI_PASS   = "ub-hs-0211";      // Wi-Fi password
const char* MQTT_SERVER = "test.mosquitto.org";
const int   MQTT_PORT   = 1883;
const char* MQTT_TOPIC  = "ub-traffic-light/signals";

uint8_t peerAddress[] = {0x88, 0x57, 0x21, 0xAD, 0x7C, 0x34};   // Receiver MAC
// --------------------------------

WiFiClient espClient;
PubSubClient client(espClient);
String clientID;   // unique MQTT client ID

// --- ESP-NOW send callback ---
void onSent(const wifi_tx_info_t*, esp_now_send_status_t status) {
  Serial.printf("[ESP-NOW] TX → %s\n",
                status == ESP_NOW_SEND_SUCCESS ? "SUCCESS" : "FAIL");
}

// --- MQTT message callback → forward to ESP-NOW ---
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  char msg[64];
  memcpy(msg, payload, length);
  msg[length] = '\0';

  Serial.printf("[MQTT] %s\n", msg);

  // Forward via ESP-NOW
  esp_err_t result = esp_now_send(peerAddress, (uint8_t*)msg, length + 1);
  Serial.printf("[ESP-NOW] Forward result: %d\n", result);
}

// --- MQTT connection helper ---
void connectMQTT() {
  if (!client.connected()) {
    Serial.print("[MQTT] Connecting...");
    if (client.connect(clientID.c_str())) {
      Serial.println("connected!");
      client.subscribe(MQTT_TOPIC);
      Serial.printf("[MQTT] Subscribed to %s\n", MQTT_TOPIC);
    } else {
      Serial.printf("failed, rc=%d\n", client.state());
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== ESP-NOW + MQTT TRANSMITTER (Stable) ===");

  // 1️⃣ Connect Wi-Fi (2.4 GHz only)
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true, true);
  delay(100);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("[Wi-Fi] Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\n[Wi-Fi] Connected, IP: %s | MAC: %s\n",
                WiFi.localIP().toString().c_str(),
                WiFi.macAddress().c_str());

  // ⭐ Generate unique MQTT Client ID
  clientID = "esp32-";
  clientID += WiFi.macAddress();
  clientID.replace(":", "");
  Serial.printf("[MQTT] Using Client ID: %s\n", clientID.c_str());

  // 2️⃣ Init ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("[ESP-NOW] Init failed!");
    while (true) delay(1000);
  }
  esp_now_register_send_cb(onSent);

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, peerAddress, 6);
  peerInfo.channel = 0;      // auto-follow Wi-Fi channel
  peerInfo.encrypt = false;
  if (esp_now_add_peer(&peerInfo) == ESP_OK)
    Serial.println("[ESP-NOW] Peer added");

  // 3️⃣ Init MQTT
  client.setServer(MQTT_SERVER, MQTT_PORT);
  client.setCallback(mqttCallback);
  connectMQTT();
}

void loop() {
  client.loop();   // keep MQTT alive

  static unsigned long lastTry = 0;
  if (!client.connected() && millis() - lastTry > 5000) {
    lastTry = millis();
    connectMQTT();
  }
}
