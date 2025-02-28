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

    # create a uploader for image
    uploaded_file = st.file_uploader("Upload a file", type=['jpg', 'jpeg', 'png'])
    # st.button("Detect", type="secondary", icon="✨")

    if uploaded_file and not st.session_state.get("diagnosis_done", False):
        # Show original image
        caption = uploaded_file.name
        image_bytes_data = BytesIO(uploaded_file.getvalue())
        st.image(image=image_bytes_data, caption=caption, channels='RGB')

        image_bytes_data.seek(0)
        image = Image.open(image_bytes_data).convert('RGB')
        image_array = np.array(image)
        
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
        st.image(image=img_bbox, caption=diagnosis, channels='RGB')        

        # 以診斷結果詢問 LLM 問診（LLAMA3.2）
        diagnosis_prompt = f"根據以下X光診斷結果，請問病人可能需要進一步哪些問診或處置？\n{diagnosis}"
        selected_model = st.session_state.get("selected_model", "llama3.2")
        llm = ChatOllama(model=selected_model, temperature=0.7)
        llm_output = llm.stream(diagnosis_prompt)
        with st.chat_message("ai"):
            llm_answer = st.write_stream(llm_output)

        # 將 LLM 回答插入 chat_history 作為第一則對話訊息
        st.session_state["chat_history"].insert(0, AIMessage(llm_answer))

        # 將診斷結果、信心度與建議事項整合成 dataframe 表格
        data = {
            "診斷結果": [diagnosis_result],
            "信心度": [f"{conf_value:.2f}"],
            "建議事項": [llm_answer]
        }
        df = pd.DataFrame(data)
        st.dataframe(df)

        # 設定診斷已處理的旗標，避免重複處理
        st.session_state["diagnosis_done"] = True

    chat_history = st.session_state["chat_history"]
    print(chat_history)
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
        if model["model"] == "llama3.2:latest":
            models_list.append(model["model"])
    
    return models_list


def run():
    st.set_page_config(page_title="Chat Application")
    st.header("X Ray Chat Application")
    st.selectbox("Select A LLM:", get_models(), key="selected_model")

    # 執行圖片上傳與診斷流程，並將 LLM 回答顯示在第一則訊息中
    app_session_init()

    # 後續可讓使用者進一步輸入訊息
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