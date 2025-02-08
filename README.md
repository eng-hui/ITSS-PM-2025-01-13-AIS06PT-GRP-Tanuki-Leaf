# Tanuki Project

## Overview
This project uses OpenCV, MediaPipe, and other libraries to capture and process hand gestures. The application includes a graphical user interface (GUI) built with Tkinter.

## Prerequisites
- Anaconda installed on your system.

## Setup

### Step 1: Create the Conda Environment
1. Ensure you have the `environment.yaml` file in the root directory of your project. The file should look like this:

    ```yaml
    name: tanuki
    channels:
      - defaults
      - conda-forge
      - pytorch
    dependencies:
      - python=3.10
      - numpy
      - pillow
      - pyyaml
      - pytorch
      - torchvision
      - torchaudio
      - cudatoolkit=11.3
      - pip
      - pip:
          - opencv-python
          - mediapipe
          - msvc-runtime
    ```

2. Open a terminal and navigate to the directory containing the environment.yaml file.

3. Run the following command to create the conda environment:

    ```sh
    conda env create -f environment.yaml
    ```

4. Activate the environment:

    ```sh
    conda activate tanuki
    ```

### Step 2: Run the Application
1. Navigate to the src directory:

    ```sh
    cd src
    ```

2. Run the main application:

    ```sh
    python main.py
    ```

## Usage
The application provides a GUI for capturing and processing hand gestures. Use the buttons and controls in the GUI to interact with the application.

## Troubleshooting
If you encounter any issues, ensure that all dependencies are installed correctly and that the conda environment is activated.

## License
This project is licensed under the MIT License.