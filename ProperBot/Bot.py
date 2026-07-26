from tempfile import template
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import os


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    temperature=0.8
)
model = ChatHuggingFace(llm = llm)
chat_history = []
while True:
    user = input("Your Query :")
    chat_history.append(user)
    if user == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print("Ai Response : ",result.content)

print(chat_history)
