---

## PROJECT TITLE  
## Tanuki Leaf: Real-Time Diffusion Avatar Generation   

![Tanuki Leaf Logo](image/logo.png)

---
## Executive Summary

### Overview

Content creators, livestreamers, and digital entertainers often face high technical and creative barriers when building personalised avatars for interactive media. Traditional avatar creation is time-consuming, requiring specialised tools and coordination between modellers, riggers, and streamers. Moreover, existing systems often lack seamless integration with gesture and speech control, limiting real-time responsiveness and user engagement.

**Tanuki Leaf** is an AI-powered solution that transforms webcam feeds into animated, stylised avatars using real-time diffusion models and gesture recognition. By combining natural language prompts, hand gestures, and speech commands, it allows users to create and control expressive avatars in real time, reducing onboarding time and unlocking creative possibilities for digital identity and interaction.

## Solution Benefits

Tanuki Leaf offers an innovative solution with the following capabilities:

- **Real-time avatar generation** using diffusion models trained via LoRA and DreamBooth, ensuring consistent and personalised outputs from simple webcam feeds.
- **Gesture-based interaction** through hand tracking and classification, allowing users to trigger avatar actions and changes with intuitive hand movements.
- **Voice-activated control** via automatic speech recognition (ASR), supporting hands-free switching of avatars and commands.
- **Seamless streaming integration** with platforms like OBS, enabling creators to broadcast dynamic avatars alongside live content on Twitch, YouTube, and more.

---

## Market Landscape and Opportunity

In Singapore’s digitally connected and innovation-driven landscape, content creation and interactive streaming are rapidly growing. Tanuki Leaf addresses the needs of a new generation of digital storytellers and performers by reducing the complexity of avatar-based engagement. Its lightweight, modular design allows it to run on consumer-grade hardware and integrate into existing broadcasting setups.

While initially scoped for local creators, the technology has strong potential for global adoption in gaming, VTubing, education, and virtual events—anywhere users need a personalised, expressive digital presence without the traditional production overhead.

---

## Conclusion and Future Prospects

Tanuki Leaf aims to evolve into a plug-and-play avatar toolkit that lowers technical barriers and enhances real-time user interaction. Future development will focus on improving avatar motion consistency, expanding gesture vocabularies, enabling facial tracking, and refining UI/UX for broader accessibility. By streamlining avatar generation into a real-time, user-friendly pipeline, Tanuki Leaf empowers creators to build authentic digital identities and engage their audiences more dynamically and intuitively.

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
