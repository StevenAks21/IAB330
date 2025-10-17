#include <ArduinoBLE.h>
#include <Arduino_LSM6DS3.h>
#include <string.h>

#define SAMPLE_HZ 50
#define SERVICE_UUID        "12345678-1234-5678-1234-56789abcdef0"
#define CHARACTERISTIC_UUID "12345678-1234-5678-1234-56789abcdef1"

BLEService imuService(SERVICE_UUID);
BLECharacteristic imuChar(CHARACTERISTIC_UUID, BLERead | BLENotify, 16);

uint32_t lastSampleMs = 0;
const uint32_t sampleIntervalMs = 1000 / SAMPLE_HZ;

void packAndNotify(uint32_t ts, float ax_g, float ay_g, float az_g, float gx_dps, float gy_dps, float gz_dps) {
  int16_t ax = (int16_t)(ax_g * 1000.0f);
  int16_t ay = (int16_t)(ay_g * 1000.0f);
  int16_t az = (int16_t)(az_g * 1000.0f);
  int16_t gx = (int16_t)(gx_dps * 1000.0f);
  int16_t gy = (int16_t)(gy_dps * 1000.0f);
  int16_t gz = (int16_t)(gz_dps * 1000.0f);

  uint8_t buf[16];
  buf[0] = (uint8_t)(ts & 0xFF);
  buf[1] = (uint8_t)((ts >> 8) & 0xFF);
  buf[2] = (uint8_t)((ts >> 16) & 0xFF);
  buf[3] = (uint8_t)((ts >> 24) & 0xFF);
  int16_t vals[6] = {ax, ay, az, gx, gy, gz};
  memcpy(&buf[4], vals, 12);
  imuChar.setValue(buf, sizeof(buf));
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) { }

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU (LSM6DS3).");
    while (1) { delay(1000); }
  }

  if (!BLE.begin()) {
    Serial.println("BLE start failed.");
    while (1) { delay(1000); }
  }

  BLE.setLocalName("Nano33-Steven");
  BLE.setDeviceName("Nano33-Steven");
  BLE.setAdvertisedService(imuService);
  imuService.addCharacteristic(imuChar);
  BLE.addService(imuService);

  uint8_t zero[16] = {0};
  imuChar.setValue(zero, 16);

  BLE.advertise();
  Serial.println("BLE advertising. Ready.");
}

void loop() {
  BLEDevice central = BLE.central();
  if (central && central.connected()) {
    if (millis() - lastSampleMs >= sampleIntervalMs) {
      lastSampleMs = millis();
      float ax, ay, az, gx, gy, gz;
      if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
        IMU.readAcceleration(ax, ay, az);
        IMU.readGyroscope(gx, gy, gz);
        packAndNotify(lastSampleMs, ax, ay, az, gx, gy, gz);
      }
    }
  } else {
    delay(10);
  }
}