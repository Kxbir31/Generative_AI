from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
st.header('Research Tool')
user_input = st.text_input('Enter your Prompt')
import os

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    temperature=0.8
)
model = ChatHuggingFace(llm = llm)
if st.button ('Summarise'):
    result = model.invoke(user_input)
    st.write(result.content)