# Bone-Fracture Identification System

This project is a system of YOLO-based for bone fracture detection and LLM for non-medical advices aimed at helping doctor's diagnosis. The system is built by `streamlit` package and `ultralytics` and `unsloth` for YOLO5 and LLM fine-tuninng respectively.

<div style="display: flex; justify-content: center; margin: 0 auto;">
  <img src="https://github.com/Leohoji/dog-cat-breed-classification-system/blob/main/README_Images/home_page.png?raw=true" alt="Home Page" style="width: 630px; height: 375px;"/>
</div>

# How do I complete this project?

✨ Watch This ▶▶▶ Here is my [System Architecture Design Specification](https://beryl-scapula-b22.notion.site/System-Architecture-Design-Specification-4d8a6fc21b8647a88fa45838d516fd20?pvs=4).

This project follows the [SDLC V-Model](https://www.geeksforgeeks.org/software-engineering-sdlc-v-model/) software development process, which includes:

1. **Requirement Analysis**: Define system functionality and performance requirements.
2. **System Design**: Design the overall architecture and database structure.
3. **Implementation**: Write code to implement functionalities.
4. **Unit Testing**: Test individual modules to ensure correctness.
5. **Integration Testing**: Test the integration between modules.
6. **Acceptance Testing**: Perform comprehensive testing in the final environment.
7. **Maintenance**: Continuously improve the system based on user feedback.

## How do I train the model?

✨ How Do I Train My Model ▶▶▶ The deep neural network training for cats or dogs classification is [HERE](https://github.com/Leohoji/dog-cat-breed-classification-system/tree/main/model_training).

## Tech Stack

- **Language**: Python 3.8
- **Framework**: Django 2.2.28
- **Deep Learning**: TensorFlow 2.10.0 
- **Frontend**: HTML, CSS, JavaScript

**GPU Available** (optional) 【💻 My GPU is **GeForce GTX 1050 Ti**】

If you have GPU to accelerate, devices I installed:

-  tensorflow-gpu 2.10.0
-  CUDA 11.2
-  cuDNN 8.1

## Features

### Login and Sign up

User have to login the website first, if you do not have an account, you can register first.

<img src="https://github.com/Leohoji/dog-cat-breed-classification-system/blob/main/README_Images/Login%20and%20Sign-up.jpg?raw=true" alt="User Login" style="width: 500px; height: 400px;"/>

### Upload Image For Classification

User can upload their animal image for breed classification, in addition, users can also view the recognition result records of this account through button "Check historical data".

<div style="display: flex; justify-content: space-around; width: 1000px">
  <img src="https://github.com/Leohoji/dog-cat-breed-classification-system/blob/main/README_Images/upload_img.png?raw=true" alt="Image Uploaded" style="width: 800px; height: 500px;"/>
</div>

<div style="display: flex; justify-content: space-around; width: 500px">
  <img src="https://github.com/Leohoji/dog-cat-breed-classification-system/blob/main/README_Images/Show%20Historical%20Data.jpg?raw=true" alt="Historical Data" style="width: 500px; height: 800px;"/>
</div>


### Classification Result

After image uploading and classifying, the classification result will display to user, the information include:

1. Detect there is an animal present on the image, and what is the animal is (cat or dog)
2. Recognize the `species` and `breed` to interface
3. Show the relative information and link to user
4. Display another two image to let user check the result
5. Provide a `feedback` area to user for record and further model fine-tune

<img src="https://github.com/Leohoji/dog-cat-breed-classification-system/blob/main/README_Images/classification_result.png?raw=true" alt="Classification Result" style="width: 800px; height: 600px;"/>

<ins>Password Forgotting?</ins>

User can enter their Gmail account to receive a verification code for password reset:

1. Collect user's Gmail
2. Generate a verification code to user
3. Verify code and password reset

<div style="display: flex; justify-content: space-around; width: 950px">
  <img src="https://github.com/Leohoji/dog-cat-breed-classification-system/blob/main/README_Images/enter-user-gmail.png?raw=true" alt="Enter User's Gmail" style="width: 250px; height: 350px;"/>
  <img src="https://github.com/Leohoji/dog-cat-breed-classification-system/blob/main/README_Images/send-verification-code.png?raw=true" alt="Send Verification Code To User" style="width: 250px; height: 350px;"/>
  <img src="https://github.com/Leohoji/dog-cat-breed-classification-system/blob/main/README_Images/reset-user-password.png?raw=true" alt="Rest Password" style="width: 250px; height: 350px;"/>
</div>


## Installation and Usage

1. Create and activate conda environment 
   ```bash
   conda create --name CatDogClassification python=3.8
   conda activate CatDogClassification
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
   
## Learning system
To be continue...

You will fine-tune the model to get better performance if the user's photo is large enough.

# Project Author

*Ho Lo*

