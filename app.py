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
model = YOLO('models/runs_v5_detect_classification/detect/train/weights/best.pt')

def app_session_init():
    # Chat history initialization
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Ensure the initialization of diagnosis flag
    if "diagnosis_done" not in st.session_state:
        st.session_state["diagnosis_done"] = False
    
    # Model initialization
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = "llama3.2"

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

    # LLM initialization
    selected_model = st.session_state["selected_model"]
    llm = ChatOllama(model=selected_model, temperature=0.7)

    return llm

def process_image(uploaded_file):

        image_bytes_data = BytesIO(uploaded_file.getvalue())
        original_image = Image.open(image_bytes_data).convert('RGB')
        image_array = np.array(original_image)
        
        # X-ray fracture detection
        results = model(image_array)

        # Access class names
        class_id, class_name, conf = '', '', ''
        for r in results:
            boxes = r.boxes  # Boxes object for bounding box outputs
            for box in boxes:
                class_id = box.cls  # Get class index
                class_name = r.names[int(class_id)]  # Get class name from names dictionary
                conf = box.conf  # Get confidence scores

        img_bbox = results[0].plot()
        diagnosis_result = 'fracture' if class_name == 'positive' else 'normal'
        conf_value = conf.cpu().item() if hasattr(conf, 'cpu') else float(conf)
        diagnosis = f"Diagnosis: {diagnosis_result} || Confidence: {conf_value:.2f}"

        # Save processed results into session_state
        st.session_state["image_data"] = original_image
        st.session_state["diagnosis_img"] = Image.fromarray(img_bbox)
        st.session_state["diagnosis_result"] = diagnosis_result
        st.session_state["diagnosis_confidence"] = conf_value
        st.session_state["diagnosis_text"] = diagnosis

        return diagnosis

def display_saved_image_and_diagnosis():
    # Only display when there is saved image data in the session
    if st.session_state["image_data"] is not None:
        # Display original image
        st.image(st.session_state["image_data"], caption="Original X-ray", channels='RGB')
        
        # Display marked diagnosis image
        if st.session_state["diagnosis_img"] is not None:
            st.image(st.session_state["diagnosis_img"], 
                     caption=f"Diagnosis: {st.session_state['diagnosis_result']} || Confidence: {st.session_state['diagnosis_confidence']:.2f}", 
                     channels='RGB')
        
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
    st.set_page_config(page_title="Chat Application")
    st.header("X Ray Chat Application")
    
    # Initialize LLM and session state
    llm = app_session_init()

    # Create three blocks: image upload, image show, and chat block
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload an X-ray image", type=['jpg', 'jpeg', 'png']) # Image upload block

        # Process image and diagnosis
        if uploaded_file and not st.session_state.get("diagnosis_done", False):
            # Process image and get diagnosis
            diagnosis = process_image(uploaded_file)

            # Ask LLM based on diagnosis results
            diagnosis_prompt = f"Based on the following X-ray diagnosis, what further examination or treatment might the patient need? Please provide a short answer (10 words max). \n{diagnosis}"
            
            # Add diagnosis question to chat history
            st.session_state["chat_history"] = [HumanMessage(diagnosis_prompt)]

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
            
            # Save LLM suggestion for display in summary
            st.session_state["llm_suggestion"] = llm_answer

            # Directly display diagnosis image and summary
            display_saved_image_and_diagnosis()

            # Set flag indicating diagnosis has been processed
            st.session_state["diagnosis_done"] = True

            # Force a rerun to refresh the UI with chat history
            # st.rerun()
    
        # If diagnosis has be done, display the chat history
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