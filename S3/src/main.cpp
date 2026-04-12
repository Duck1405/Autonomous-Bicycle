#include <Arduino.h>
#include <Wire.h>
#include <Servo.h>
#include "pins.h"
#include <AS5600.h>


AS5600 encoder;
Servo esc;


bool systemActive = false;
bool manualOverride = false;
int targetSpeed = 0;
float manualAngle = 0.0;
float remoteAngle = 0.0;

unsigned long lastClickTime = 0;
int clickCount = 0;


void handleI2C(int bytes);
void processSerial();
float readAngle();

void setup() {
  Serial.begin(9600);

  Wire.begin(I2C_SDA, I2C_SCL, 0x04);
  Wire.onReceive(handleI2C);

  pinMode(PIN_RELAY_L, OUTPUT); //left
  pinMode(PIN_RELAY_R, OUTPUT); //right
  pinMode(PIN_RELAY_BRAKE, OUTPUT); //brake
  digitalWrite(PIN_RELAY_L, HIGH); // Default OFF
  digitalWrite(PIN_RELAY_R, HIGH); // Default OFF
  digitalWrite(PIN_RELAY_BRAKE, HIGH); // Default OFF

  ledcSetup(PWM_CHAN_SPEED, PWM_FREQ, PWM_RES); //basic settings
  ledcAttachPin(PIN_PWM_SPEED, PWM_CHAN_SPEED); // Op amp cpnection for speed

  esc.attach(PIN_ESC_SIGNAL); // ESC initialization
  esc.writeMicroseconds(1500); // Arming signal


  encoder.begin();
  
  Serial.println("System Started. Awaiting commands...");



}

void loop() {
  processSerial();
  float currentAngle = readAngle();

  float activeAngle;
  if (manualOverride) {
    activeAngle = manualAngle;
  } 
  else {
    activeAngle = remoteAngle;
  }



  if (systemActive) {
    // 1. BRAKE LOGIC: If brakes are on, kill speed
    if (digitalRead(PIN_RELAY_BRAKE) == LOW) { // Assuming LOW is trigger
        ledcWrite(PWM_CHAN_SPEED, 0);
        esc.writeMicroseconds(1500);
    } else {
        // 2. SPEED CONTROL: Write targetSpeed to Op-Amp
        ledcWrite(PWM_CHAN_SPEED, targetSpeed);

        // 3. STEERING LOGIC: Simple Proportional Control
        // If angle is off by more than 2 degrees, move ESC
        float error = activeAngle - currentAngle;
        if (abs(error) > 2.0) {
            int escSignal = 1500 + (error * 10); // Simple gain of 10
            escSignal = constrain(escSignal, 1200, 1800);
            esc.writeMicroseconds(escSignal);
        } else {
            esc.writeMicroseconds(1500); // Stop at target
        }
    }
  } 
  
  else {
    // System IDLE
    ledcWrite(PWM_CHAN_SPEED, 0);
    esc.writeMicroseconds(1500);
  }

  // Monitor Output
  static unsigned long timer = 0;
  if (millis() - timer > 200) {
    Serial.printf("Mode: %s | Ang: %.1f | Target: %.1f | Spd: %d\n", 
                  systemActive ? "ON" : "OFF", currentAngle, activeAngle, targetSpeed);
    timer = millis();
  }
}



float readAngle() {
  // Map 0-360 to -45 to +45 (requires calibration based on your mount)
  float raw = encoder.readAngle() * (360.0 / 4096.0);
  float deg = raw - 180.0; // Assume 180 is center
  return constrain(deg, -45.0, 45.0);
}

void processSerial() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    if (input.startsWith("angle_remote ")) {
      remoteAngle = input.substring(13).toFloat();
    }
  }
}

void handleI2C(int bytes) {
  String cmd = "";
  while (Wire.available()) cmd += (char)Wire.read();

  if (cmd == "BTN:TRIPLE") systemActive = !systemActive;
  else if (cmd == "BTN:HOLD") manualOverride = true;
  else if (cmd == "BTN:RELEASE") manualOverride = false;
  else if (cmd.startsWith("SPD:")) targetSpeed = cmd.substring(4).toInt();
  else if (cmd.startsWith("ANG:")) manualAngle = cmd.substring(4).toFloat();
}




