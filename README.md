# 📌 Project Overview

This repository hosts the code, documentation, and development work for the Autonomous Bicycle Project, powered by:

- Jetson Nano Super (AI processing)
- ESP32-S3 (motor + steering controller)
- 250W motor system
- Sensors: camera, ultrasonic, IMU, etc.

The goal of this GitHub is to keep all AI, hardware, and design work organized, maintainable, and easy for collaborators to understand.

## 📂 Repository Directory Structure
1. **main** branch
   Purpose:
     Documentation only — no code.
   Contains:
     README.md
     High-level Docs (Hosted on Vercel??)

2. **ai** branch
   Purpose:
     All machine learning, AI, and Jetson work.
   Contains:
     Computer vision models
     YOLO, OpenCV scripts
     AI training notebooks
     Data preprocessing scripts
     Jetson inference code
   Rules:
     Use Git LFS for large files
     Must follow a consistent folder structure

3. **hardware** branch
   Purpose:
     Everything that runs on the ESP32-S3 or interacts with electronics.
   Contains:
     Motor control firmware
     Steering servo code
     Sensor drivers
     IMU drivers
     Serial/UART protocols

## 📌 Branch Merging Rules
- _ai_ → merged only after accuracy/performance tested
- _hardware_ → merged only after firmware compiles
- _docs_ → merged after visual check

## 📌 Code Quality Requirements
- No unused files
- Comment major functions
- Must include test cases when possible
- Keep folders clean and organized







   
