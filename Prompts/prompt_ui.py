import os
from dotenv import load_dotenv
import streamlit as st

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

load_dotenv()



# ==========================================
# STREAMLIT UI
# ==========================================

st.title("🤖 GenAI Chatbot")


# ==========================================
# 3 DROPDOWNS
# ==========================================

role = st.selectbox(
    "Choose Role",
    [
        "Teacher",
        "Software Engineer",
        "Interviewer"
    ]
)

topic = st.selectbox(
    "Choose Topic",
    [
        "Machine Learning",
        "Generative AI",
        "Python"
    ]
)

difficulty = st.selectbox(
    "Choose Difficulty",
    [
        "Beginner",
        "Intermediate",
        "Advanced"
    ]
)


# ==========================================
# CONVERT SELECTED OPTIONS INTO VARIABLES
# ==========================================

role_variable = role
topic_variable = topic
difficulty_variable = difficulty



# ==========================================
# HUGGING FACE MODEL
# ==========================================

hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    temperature=0.7,
    huggingfacehub_api_token=hf_token
)

chat_model = ChatHuggingFace(llm=llm)

template = load_prompt('template.json')

# BUTTON
if st.button("Generate Answer"):

    # --------------------------------------
    # STEP 1: PromptTemplate.invoke()
    # --------------------------------------

    prompt = template.invoke({
        "role": role_variable,
        "topic": topic_variable,
        "difficulty": difficulty_variable
    })


    # --------------------------------------
    # STEP 2: ChatModel.invoke()
    # --------------------------------------

    response = chat_model.invoke(prompt)


    # --------------------------------------
    # STEP 3: Display Result
    # --------------------------------------

    st.subheader("AI Response")
    st.write(response.content)