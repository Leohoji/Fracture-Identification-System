import sys
import numpy as np
from PIL import Image
from io import BytesIO

import ollama
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama

from ultralytics import YOLO
model = YOLO('models/runs_v5_detect_classification/detect/train/weights/best.pt')

def app_session_init():
    # create a uploader for image
    uploaded_file = st.file_uploader("Upload a file", type=['jpg', 'jpeg', 'png'])
    # st.button("Detect", type="secondary", icon="✨")
    
    if uploaded_file:
        # Show original image
        caption = uploaded_file.name
        image_bytes_data = BytesIO(uploaded_file.getvalue())
        st.image(image=image_bytes_data, caption=caption, channels='RGB')

        image_bytes_data.seek(0)
        image = Image.open(image_bytes_data).convert('RGB')
        image_array = np.array(image)
        print(image_array.shape)
        
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
        diagnosis = f"Diagnosis: {('fracture' if class_name=='positive' else 'normal')} || Confidence: {conf.cpu().numpy()[0]:.2f}"
        st.image(image=img_bbox, caption=diagnosis, channels='RGB')        

    # create a session for chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [AIMessage("Hello, how can I help you?")]
    
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = "llama3.2"

    chat_history = st.session_state["chat_history"]
    for history in chat_history:
        if isinstance(history, AIMessage):
            st.chat_message("ai").write(history.content)
        
        if isinstance(history, HumanMessage):
            st.chat_message("human").write(history.content)

def get_models():
    models = ollama.list()
    if not models:
        print("No models found, please visit: https://ollama.dev/models")
        sys.exit(1)

    models_list = []
    for model in models["models"]:
        models_list.append(model["model"])
    
    return models_list


def run():
    st.set_page_config(page_title="Chat Application")
    st.header("X Ray Chat Application")
    st.selectbox("Select A LLM:", get_models(), key="selected_model")

    app_session_init()
    prompt = st.chat_input("Add your prompt...")

    selected_model = st.session_state["selected_model"]
    print(f"Selected Model: {selected_model}")
    llm = ChatOllama(model=selected_model, temperature=0.7)

    if prompt:
        st.chat_message("user").write(prompt)
        st.session_state["chat_history"] += [HumanMessage(prompt)]
        output = llm.stream(prompt) # an iterator for continuous talking

        with st.chat_message("ai"):
            ai_message = st.write_stream(output) # iterator
        
        st.session_state["chat_history"] += [AIMessage(ai_message)]


if __name__ == "__main__":
    run()