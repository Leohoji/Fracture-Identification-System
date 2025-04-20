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
   conda create --name xrayenv python=3.8
   conda activate xrayenv
   ```
2. Clone and enter this repository:
   ```bash
   git clone https://github.com/Leohoji/dog-cat-breed-classification-system.git
   cd dog-cat-breed-classification-system
   ```
3. Set MySQL database
   Download the [MySQL](https://www.mysql.com/downloads/) data base to your local computer, and create a file named `mysql_info.py ` to save the path to `\CatDogClassification\mysql_info.py`:
   ```python
   HOST = 'localhost'
   PORT = '3306'
   USER = 'root'
   PASSWORD = '' # your password
   DATABASE_NAME = 'cat_dog_system'
   ```
4. Set Gmail app password
   Set your [Gmail app passwords](https://support.google.com/mail/answer/185833?hl=en) for receiving verification code of **password forgetting** service, and create a file named `python_mail.py` to save the
   path `\CatDogClassification\python_mail.py`:
   ```python
   Gmail_Account = '' # your Gmail account
   Gmail_Password = "" # your Gmail app password
   ```
5. Prepare dataset for learning system
   Download the dataset from [Cats and Dogs Breeds Classification Oxford Dataset](https://www.kaggle.com/datasets/zippyz/cats-and-dogs-breeds-classification-oxford-dataset) and put it to your own directory for model training.
   
7. Install required package
   ```bash
   pip install -r requirements.txt
   conda install jupyter jupyterlab
   ```
8. Run the django server
   Run the following program instructions and copy the address `http://127.0.0.1:8000/` to your local device.
   ```bash
   python manage.py runserver
   ```

# Project Author

*Ho Lo*

