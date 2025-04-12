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
# DEFAULT_MODEL = "hf.co/loholeo/medical-Phi-3.5-mini-instruct:latest"
DEFAULT_MODEL = "llama3.2:latest"
YOLO_PATH = 'C:/Users/User/Desktop/bone_fracture_detection_project/XRayDetection/yolov5/weights/best.pt'

def app_session_init():
    # Initialize basic state variables using setdefault
    st.session_state.setdefault("chat_history", []) # chat history
    st.session_state.setdefault("diagnosis_done", False) # diagnosis flag
    st.session_state.setdefault("selected_model", DEFAULT_MODEL) # LLM model
    st.session_state.setdefault("diagnosis_result", None)
    st.session_state.setdefault("diagnosis_confidence", None)
    st.session_state.setdefault("diagnosis_img", None)
    
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
    class_id, class_name = '', ''
    confidences = []
    img_bbox = None

    for r in results:
        img_bbox = r.plot()
        boxes = r.boxes  # Boxes object for bounding box outputs
        for box in boxes:
            class_id = box.cls  # Get class index
            class_name = r.names[int(class_id)]  # Get class name from names dictionary
            confidences.append(box.conf.cpu().numpy())

    diagnosis_result = 'fractured' if class_name == 'Fractured' else 'normal'
    conf_value = np.mean(confidences)
    diagnosis = f"The X-ray shows that the person has a {diagnosis_result} bone with a confidence level of {conf_value:.2f}."

    # Save processed results into session_state
    st.session_state["diagnosis_img"] = Image.fromarray(img_bbox)
    st.session_state["diagnosis_result"] = diagnosis_result
    st.session_state["diagnosis_confidence"] = conf_value
    st.session_state["diagnosis_text"] = diagnosis

    return diagnosis

def display_saved_image_and_diagnosis(f_name="diagnosis"):     
    # Display marked diagnosis image
    if st.session_state["diagnosis_img"] is not None:
        st.image(st.session_state["diagnosis_img"], caption=f_name, channels='RGB')
    
    # Display diagnosis summary table
    if st.session_state["diagnosis_done"] is not None:
        st.subheader("Diagnosis Summary")
        data = {
            "Value": [
                st.session_state["diagnosis_result"],
                f"{st.session_state['diagnosis_confidence']:.2f}"
            ]
        }
        df = pd.DataFrame(data, index=["Diagnosis", "Confidence"])
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
    
    return [model["model"] for model in models["models"]]


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
                if st.session_state["selected_model"] == "llama3.2:latest":
                    diagnosis_prompt = """{} Please provide some daily living suggestions that may be helpful during recovery, such as non-medical suggestions on how to adjust to temporary mobility issues, home environment adjustments, psychological adjustments, or safe recreational activities that can be carried out during recovery. Please make it clear that you are not providing medical advice and that any recovery-related decisions should be made in consultation with a medical professional.
                    """.format(diagnosis)
                else:
                    # Ask LLM based on diagnosis results
                    diagnosis_prompt = """Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\nBefore answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.\n
                    
                    ### Instruction:
                    You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
                    Please answer the following medical question, output a response for detail solution and a 50 words summarization.

                    ### Question:
                    {}

                    ### Response:
                    {}

                    ### Summary:
                    {}""".format(diagnosis, "", "")
                
                # Get LLM response (without displaying in UI yet)
                full_response = "" 

                # Use proper content extraction from chunks
                for chunk in llm.stream(diagnosis_prompt):
                    if hasattr(chunk, 'content'):
                        full_response += chunk.content
                    else:
                        # Fallback in case chunk format changes
                        full_response += str(chunk)
                
                # Initialize chat history with diagnosis Q&A
                st.session_state["chat_history"] = [
                    HumanMessage(diagnosis_prompt),
                    AIMessage(full_response)
                ]

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