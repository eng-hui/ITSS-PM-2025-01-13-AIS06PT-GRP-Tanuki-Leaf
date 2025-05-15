---

## PROJECT TITLE  
## Tanuki Leaf: Real-Time Diffusion Avatar Generation   

![Tanuki Leaf Logo](image/logo.png)

---
## Executive Summary

### Overview

Tanuki Leaf is an intelligent, real-time avatar generation system built for streamers, gamers, and digital creators. The project addresses the challenges of high entry barriers, fragmented workflows, and limited personalisation found in traditional avatar creation tools. Our solution leverages AI-powered tools to create avatars driven by webcam input, natural language prompts, and hand gestures.

### Key Features

- **Real-Time Avatar Generation**: Uses LoRA-fine-tuned diffusion models to create consistent, stylised avatars from live webcam feeds.
- **Gesture-Based Control**: Integrates Mediapipe and traditional ML models to recognise hand gestures and trigger avatar behaviours.
- **Prompt Personalisation**: Enables detailed visual control of avatars using structured, weighted natural language prompts.
- **Voice Interaction**: Implements speech recognition to allow hands-free control over avatar switching.
- **Streaming Integration**: Seamlessly combines avatar visuals with live gameplay through OBS for platforms like Twitch and YouTube.

### Technical Highlights

- **Diffusion Engine**: StreamDiffusion framework with integrated ControlNet for pose consistency and low-latency performance.
- **LoRA & DreamBooth**: Fine-tunes avatar appearance efficiently using small datasets for rapid personalisation.
- **Machine Learning Models**: Evaluated YOLO and Mediapipe for hand tracking; trained classifiers (e.g., Decision Tree) for gesture recognition.
- **Prompt Engineering**: Achieved high visual consistency with weighted token prompts and modular LoRA injection.

### Outcomes

- Reduced avatar creation time from weeks to minutes.
- Achieved usable frame rates (10–20+ FPS) on consumer-grade hardware.
- Delivered a fully functional MVP with real-time responsiveness, gesture recognition, and voice control.
- Demonstrated streaming-ready integration with platforms via OBS.

### Future Directions

- Enhance temporal consistency and avatar stability during continuous rendering.
- Expand gesture vocabulary and incorporate facial tracking for richer interaction.
- Develop a web-based, lightweight version for broader accessibility.
- Automate prompt generation using adaptive, user-feedback-driven systems.

Tanuki Leaf exemplifies the fusion of cutting-edge AI and user-centric design, empowering content creators with accessible, dynamic, and interactive digital identities.

---

## PROJECT TEAM MEMBERS

| Official Full Name  | Student ID | Email  |
| :------------ |:---------------:| :-----|
| Tan Eng Hui | A0291201W | e1330340@u.nus.edu |
| Hu Lei | A0120681N | e1329735@u.nus.edu |
| Wang Tao | A0291189R | e1330328@u.nus.edu |
| Ho Zi Hao Timothy| A0150123B | e0015027@u.nus.edu |

## PROJECT REPORT

Refer to [Project Report.pdf](ProjectReport/Project%20Report.pdf) in the **ProjectReport** folder

---

## Setup

### Prerequisites
- Anaconda installed on your system.
- GPU (CUDA) supported e.g. NVIDIA RTX 3060~4080 

### Step 1: Create the Conda Environment
1. Ensure you have the `environment.yaml` file in the root directory of your project. 

2. Open a terminal and navigate to the directory containing the environment.yaml file.

3. Run the following command to create the conda environment:

    ```sh
    conda env create -f environment.yaml
    ```

4. Activate the environment:

    ```sh
    conda activate tanuki
    ```
5. Install streamdiffusion tensorrt
    ```sh
    python -m streamdiffusion.tools.install-tensorrt
    ```

### Step 2: Run the Application
1. Run the main application:

    ```sh
    python -m src.main
    ```
