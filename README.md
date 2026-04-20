EW309 Autonomous SENTRY Turret

This repository contains the vision processing and micro-controller code for a fully autonomous, 2-axis targeting turret. Designed as the final capstone project for EW309, this system uses an OAK-D spatial AI camera and a custom YOLOv11 model to actively track, lock onto, and engage specific targets while filtering out decoys.
⚙️ System Architecture

The project is split into two primary components that communicate via a 30Hz Serial link:

1. The Commander (Host PC / Python)

    Vision Pipeline: Utilizes the DepthAI v3 API to stream 640x480 video while simultaneously running hardware-accelerated YOLOv11 inference on the OAK-D's VPU.

    Raw Tensor Decoding: Bypasses legacy API limitations by pulling raw FP16 byte arrays directly from the camera chip, using numpy and OpenCV for host-side anchor-free bounding box decoding.

    Ballistic Math: Translates pixel error into angular degrees using custom camera intrinsics (FX/FY) and physical offset biases.

    Autonomous State Machine: Includes a pre-calculation HUD, "anti-blur" target memory, strict size/color decoy filtering, and sequential multi-target engagement.

2. The Soldier (Raspberry Pi Pico / CircuitPython)

    PID Visual Servoing: A heavily damped, custom PID controller designed to eliminate mechanical oscillation and steady-state error.

    Hardware Interfacing: Drives dual PWM pan/tilt motors and digital IO for the SENTRY gun's loader and launcher mechanisms.

🎯 Performance

Built to strictly adhere to the EW309 final demonstration rubric, the system is capable of acquiring two targets, achieving a steady-state angular error of ≤0.6∘, and completing its firing sequence in under 15 seconds.
