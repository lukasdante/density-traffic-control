#include <WiFi.h>
#include <esp_now.h>
#include "esp_wifi.h"
#include <PubSubClient.h>

// ---------- CONFIG ----------
const char* WIFI_SSID   = "ub-hs-2.4G";      // 2.4 GHz Wi-Fi SSID
const char* WIFI_PASS   = "ub-hs-0211";      // Wi-Fi password
const char* MQTT_SERVER = "broker.hivemq.com";
const int   MQTT_PORT   = 1883;
const char* MQTT_TOPIC  = "ub-traffic-light/signals/horizontal";

uint8_t peerAddress[] = {0x88, 0x57, 0x21, 0xAD, 0x7C, 0x34};   // Receiver MAC
// -------------

// ---------- PINS & STATE ----------
#define RED_PIN    27
#define YELLOW_PIN 26
#define GREEN_PIN  25
#define YELLOW_SECONDS 3
#define MAX_GREEN_SECONDS 90
// --------------------------------

WiFiClient espClient;
PubSubClient client(espClient);
String clientID;   // unique MQTT client ID
enum Color { RED, YELLOW, GREEN };
void handleTrafficCommand(const char* command);
unsigned long greenStart = 0;
bool isGreen = false;


// --- Set LEDs according to color ---
void setLight(Color color) {
  digitalWrite(RED_PIN,    color == RED    ? LOW : HIGH);
  digitalWrite(YELLOW_PIN, color == YELLOW ? LOW : HIGH);
  digitalWrite(GREEN_PIN,  color == GREEN  ? LOW : HIGH);

  // 🔹 Serial log for visual trace
  switch (color) {
    case RED:    Serial.println("[LIGHT] RED ON");    break;
    case YELLOW: Serial.println("[LIGHT] YELLOW ON"); break;
    case GREEN:  Serial.println("[LIGHT] GREEN ON");  break;
  }
}

// --- ESP-NOW send callback ---
void onSent(const wifi_tx_info_t*, esp_now_send_status_t status) {
  Serial.printf("[ESP-NOW] TX → %s\n",
                status == ESP_NOW_SEND_SUCCESS ? "SUCCESS" : "FAIL");
}

// --- ESP-NOW receive callback ---
void onReceive(const esp_now_recv_info *info, const uint8_t *data, int len) {
  char msg[64];
  memcpy(msg, data, len);
  msg[len] = '\0';
  Serial.printf("[ESP-NOW RX] From %02X:%02X:%02X:%02X:%02X:%02X | Data: %s\n",
                info->src_addr[0], info->src_addr[1], info->src_addr[2],
                info->src_addr[3], info->src_addr[4], info->src_addr[5], msg);

  // React to peer command
  if (strcmp(msg, "green") == 0 || strcmp(msg, "red") == 0) {
    handleTrafficCommand(msg);
  }
}

// --- Traffic Light Logic ---
void handleTrafficCommand(const char* command) {
  // Handles only red
  Serial.printf("[LOGIC] Handling command: %s\n", command);

  if (strcmp(command, "red") == 0){
      // Become yellow first
    setLight(YELLOW);
    delay(YELLOW_SECONDS * 1000);

    // Send green to other (this is handled by espnow callback)
    esp_now_send(peerAddress, (uint8_t*)"green", 6);
    // Become red
    setLight(RED);
    isGreen = false;
  }

  // Handles only green
  if (strcmp(command, "green") == 0) {
    // Turn to green immediately
    setLight(GREEN);
    greenStart = millis();
    isGreen = true;
  }
}

// --- MQTT message callback → forward to ESP-NOW + run traffic logic ---
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  char msg[64];
  memcpy(msg, payload, length);
  msg[length] = '\0';

  Serial.printf("[MQTT] %s\n", msg);

  // Perform light behavior
  handleTrafficCommand(msg);
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
  Serial.println("\n=== ESP-NOW + MQTT TRAFFIC LIGHT ===");

  // LED pins
  pinMode(RED_PIN, OUTPUT);
  pinMode(YELLOW_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);

    // turn all OFF initially (active LOW)
  digitalWrite(RED_PIN, HIGH);
  digitalWrite(YELLOW_PIN, HIGH);
  digitalWrite(GREEN_PIN, HIGH);

  // 1️⃣ Wi-Fi connect
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

  // ⭐ Unique MQTT Client ID
  clientID = "esp32-";
  clientID += WiFi.macAddress();
  clientID.replace(":", "");
  Serial.printf("[MQTT] Using Client ID: %s\n", clientID.c_str());

  // 2️⃣ ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("[ESP-NOW] Init failed!");
    while (true) delay(1000);
  }
  esp_now_register_send_cb(onSent);
  esp_now_register_recv_cb(onReceive);

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, peerAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  if (esp_now_add_peer(&peerInfo) == ESP_OK)
    Serial.println("[ESP-NOW] Peer added");

  // 3️⃣ MQTT
  client.setServer(MQTT_SERVER, MQTT_PORT);
  client.setCallback(mqttCallback);
  connectMQTT();
}

void loop() {
  if (isGreen && (millis() - greenStart > MAX_GREEN_SECONDS * 1000)) {
    isGreen = false;
    Serial.println("[TIMER] Max green time reached → switching to red");
    handleTrafficCommand("red");
  }

  client.loop();   // keep MQTT alive

  static unsigned long lastTry = 0;
  if (!client.connected() && millis() - lastTry > 5000) {
    lastTry = millis();
    connectMQTT();
  }
}
