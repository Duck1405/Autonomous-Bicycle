#ifndef PINS_H
#define PINS_H

#include <Arduino.h>

// --- I2C Bus for AS5600 & NodeMCU ---
// ESP32-S3 default I2C is usually 8 (SDA) and 9 (SCL), 
// but you can define custom ones here.
#define I2C_SDA 8
#define I2C_SCL 9

// --- Relay Modules ---
#define PIN_RELAY_L      4   // Left Control
#define PIN_RELAY_R      5   // Right Control
#define PIN_RELAY_BRAKE  6   // Linear Actuator Brake

// --- Actuator / Motor Control ---
#define PIN_PWM_SPEED    7   // Op-Amp Speed Control (0-5V)
#define PWM_CHAN_SPEED   0   // LEDC Channel
#define PWM_FREQ         5000 // 5kHz for smooth analog conversion
#define PWM_RES          8    // 8-bit resolution (0-255)


#define PIN_ESC_SIGNAL   18  // ESC Signal (Must be PWM capable)

// --- Built-in / Status LEDs ---
#define PIN_LED_STATUS   2   // Internal or External Status LED

#endif