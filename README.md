# Bone-Fracture Identification System

This project is a system of YOLO-based for bone fracture detection and LLM for non-medical advices aimed at helping doctor's diagnosis. The system is built by `streamlit` package and `ultralytics` and `unsloth` for YOLO5 and LLM fine-tuninng respectively.

<div style="display: flex; justify-content: center; margin: 0 auto;">
  <img src="https://github.com/Leohoji/Fracture-Identification-System/blob/main/XRay_Detection_Archtecture_Design.png?raw=true" alt="system-architecture" style="width: 650px; height: 530px;"/>
</div>

# How do I train the model?

<div style="display: flex; justify-content: center; margin: 0 auto;">
  <img src="https://github.com/Leohoji/Fracture-Identification-System/blob/main/XRay_Detection_Training_Design.png?raw=true" alt="model-training-pipeline" style="width: 900px; height: 350px;"/>
</div>

# Installation and Usage

1. Create and activate conda environment 
   ```bash
   conda create --name xrayenv python=3.10
   conda activate xrayenv
   ```
2. Clone and enter this repository:
   ```bash
   git clone https://github.com/Leohoji/Fracture-Identification-System.git
   cd Fracture-Identification-System
   ```
3. Install required package
   ```bash
   conda install jupyter jupyterlab
   pip install -r requirements.txt
   ```
8. Run the streamlit server
   ```bash
   streamlit run app.py
   ```

# Project Author

*Ho Lo*

