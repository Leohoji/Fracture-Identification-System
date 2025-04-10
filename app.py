import sys
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

import ollama
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama

from ultralytics import YOLO
DEFAULT_MODEL = "llama3.2"
# YOLO_PATH = 'models/runs_v5_detect_classification/detect/train/weights/best.pt'
# YOLO_PATH = 'C:/Users/User/Desktop/bone_fracture_detection_project/runs/detect/train6/weights/best.pt'
YOLO_PATH = 'C:/Users/User/Desktop/bone_fracture_detection_project/XRayDetection/yolov5/weights/best.pt'

def app_session_init():
    # Chat history initialization
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Ensure the initialization of diagnosis flag
    if "diagnosis_done" not in st.session_state:
        st.session_state["diagnosis_done"] = False
    
    # Model initialization
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = DEFAULT_MODEL

    # Save images and diagnosis
    if "image_data" not in st.session_state:
        st.session_state["image_data"] = None
    
    if "diagnosis_result" not in st.session_state:
        st.session_state["diagnosis_result"] = None
    
    if "diagnosis_confidence" not in st.session_state:
        st.session_state["diagnosis_confidence"] = None
    
    if "diagnosis_img" not in st.session_state:
        st.session_state["diagnosis_img"] = None

    if "llm_suggestion" not in st.session_state:
        st.session_state["llm_suggestion"] = None

    if "yolo_model" not in st.session_state:
        with st.spinner("Loading YOLO model..."):
            try:
                st.session_state["yolo_model"] = YOLO(YOLO_PATH)
            except Exception as e:
                st.error(f"Error loading YOLO model: {str(e)}")
                st.session_state["yolo_model"] = None
            
    # LLM initialization
    selected_model = st.session_state["selected_model"]
    llm = ChatOllama(model=selected_model, temperature=0.7)

    return llm

def process_image(uploaded_file):
    """Process uploaded image to bone fracture analysis."""        
    if st.session_state["yolo_model"] is None:
        st.error("YOLO model is not loaded properly. Cannot process image.")
        return None
    
    image_bytes_data = BytesIO(uploaded_file.getvalue())
    original_image = Image.open(image_bytes_data).convert('RGB')
    image_array = np.array(original_image)
    
    # X-ray fracture detection
    with st.spinner("Analyzing X-ray image..."):
        results = st.session_state["yolo_model"](image_array)

    # Access class names
    class_id, class_name, conf = '', '', ''
    for r in results:
        boxes = r.boxes  # Boxes object for bounding box outputs
        for box in boxes:
            class_id = box.cls  # Get class index
            class_name = r.names[int(class_id)]  # Get class name from names dictionary
            conf = box.conf  # Get confidence scores

    img_bbox = results[0].plot()
    diagnosis_result = 'bone-fractured' if class_name == 'positive' else 'normal'
    conf_value = conf.cpu().item() if hasattr(conf, 'cpu') else float(conf)
    diagnosis = f"My initial diagnosis {diagnosis_result}, and my confidence level is {conf_value:.2f}."

    # Save processed results into session_state
    st.session_state["image_data"] = original_image
    st.session_state["diagnosis_img"] = Image.fromarray(img_bbox)
    st.session_state["diagnosis_result"] = diagnosis_result
    st.session_state["diagnosis_confidence"] = conf_value
    st.session_state["diagnosis_text"] = diagnosis

    return diagnosis

def display_saved_image_and_diagnosis(f_name="diagnosis"):
    # Only display when there is saved image data in the session
    if st.session_state["image_data"] is not None:       
        # Display marked diagnosis image
        if st.session_state["diagnosis_img"] is not None:
            st.image(st.session_state["diagnosis_img"], caption=f_name, channels='RGB')
        
        # Display diagnosis summary table
        if st.session_state["llm_suggestion"] is not None:
            st.subheader("Diagnosis Summary")
            data = {
                "Value": [
                    st.session_state["diagnosis_result"],
                    f"{st.session_state['diagnosis_confidence']:.2f}",
                    st.session_state["llm_suggestion"]
                ]
            }
            df = pd.DataFrame(data, index=["Diagnosis", "Confidence", "Recommendations"])
            st.table(df)
        
def render_chat_history(container):
    # Render chat history to the specified container
    with container:
        for history in st.session_state["chat_history"]:
            if isinstance(history, AIMessage):
                st.chat_message("ai").write(history.content)
            elif isinstance(history, HumanMessage):
                st.chat_message("human").write(history.content)

def get_models():
    models = ollama.list()
    if not models:
        print("No models found, please visit: https://ollama.dev/models")
        sys.exit(1)

    models_list = []
    for model in models["models"]:
        if model["model"] == "llama3.2:latest":
            models_list.append(model["model"])
    
    return models_list


def run():
    st.set_page_config(
        page_title="X-Ray Fracture Detection",
        page_icon="🧊",
        layout="wide"
    )
    st.title("📊 X-Ray Fracture Detection Assistant")
    st.markdown("""
    This application analyzes X-ray images to detect bone fractures and provides medical suggestions through an AI assistant.
    """)
    
    # Initialize LLM and session state
    llm = app_session_init()

    # Sidebar for settings and information
    with st.sidebar:
        st.header("Settings")
        models = get_models()
        selected_model = st.selectbox(
            "Select LLM Model", 
            options=models,
            index=models.index(st.session_state["selected_model"]) if st.session_state["selected_model"] in models else 0
        )
        
        if selected_model != st.session_state["selected_model"]:
            st.session_state["selected_model"] = selected_model
            st.rerun()
            
        if st.button("Reset Application", type="secondary", icon="🚨"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
            
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This application uses:
        - YOLO for X-ray image analysis
        - Ollama LLM for medical advice
        - Streamlit for the user interface
        """)

    # Create three blocks: image upload, image show, and chat block
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Upload X-ray Image")
        uploaded_file = st.file_uploader("Choose an X-ray image file", type=['jpg', 'jpeg', 'png']) # Image upload block

        # Process image and diagnosis
        if uploaded_file and not st.session_state.get("diagnosis_done", False):
            # Process image and get diagnosis
            diagnosis = process_image(uploaded_file)


            with st.spinner("Getting AI Advices..."):
                # Ask LLM based on diagnosis results
                diagnosis_prompt = f"Here is the initial diagnosis from YOLO model:\n ```{diagnosis}```.\nBased on the above X-ray diagnosis, what further examination or treatment might the patient need? Please provide some non-medical advices."
                
                # Get LLM response (without displaying in UI yet)
                llm_answer = "" 

                # Use proper content extraction from chunks
                for chunk in llm.stream(diagnosis_prompt):
                    if hasattr(chunk, 'content'):
                        llm_answer += chunk.content
                    else:
                        # Fallback in case chunk format changes
                        llm_answer += str(chunk)
                
                # Initialize chat history with diagnosis Q&A
                st.session_state["chat_history"] = [
                    HumanMessage(diagnosis_prompt),
                    AIMessage(llm_answer)
                ]
                
                # summary the output of LLM's advices
                llm_suggestion = ""
                for chunk in llm.stream(f"Output the summary of the following advices in one sentences directly.\n```{llm_answer}```"):
                    if hasattr(chunk, 'content'):
                        llm_suggestion += chunk.content
                    else:
                        # Fallback in case chunk format changes
                        llm_suggestion += str(chunk)
                # Save LLM suggestion for display in summary
                st.session_state["llm_suggestion"] = llm_suggestion

                # Directly display diagnosis image and summary
                display_saved_image_and_diagnosis()

                # Set flag indicating diagnosis has been processed
                st.session_state["diagnosis_done"] = True
        
        # If diagnosis has been done, display the result
        elif st.session_state.get("diagnosis_done", False):
            display_saved_image_and_diagnosis()

    # Right area: Chat
    with col2:
       # If diagnosis is completed, display chat history
       if st.session_state.get("diagnosis_done", False):
            # Render existing chat history
            render_chat_history(col2)
            
    # Chat input box (only displayed after diagnosis is completed)
    if st.session_state.get("diagnosis_done", False):
        
        # Add chat input box at the bottom of the page
        prompt = st.chat_input("Enter your question...")
        
        if prompt:
            # User's question is added to chat history later
            # Process response and add to chat history
            # First add to chat history
            st.session_state["chat_history"].append(HumanMessage(prompt))
            
            # Display in UI
            with col2:
                with st.chat_message("human"):
                    st.write(prompt)
                
                with st.chat_message("ai"):
                    message_placeholder = st.empty()
                    ai_response = ""
                    # Use proper content extraction from chunks
                    for chunk in llm.stream(prompt):
                        if hasattr(chunk, 'content'):
                            ai_response += chunk.content
                        else:
                            # Fallback in case chunk format changes
                            ai_response += str(chunk)
                        message_placeholder.write(ai_response)
            
            # Add AI answer to chat history
            st.session_state["chat_history"].append(AIMessage(ai_response))
            
            # Force a rerun to refresh the UI with chat history
            st.rerun()

if __name__ == "__main__":
    run()