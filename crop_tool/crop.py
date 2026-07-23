from tempfile import template
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
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

st.header("AgroSense AI")
user_input = st.text_input(" Ask Me Any Thing")
state = st.text_input("Enter your state ")
district = st.text_input("Enter your district ")


template = PromptTemplate(
    input_variables=["question", "location", "district"],
    template=f"""
You are an expert Agriculture AI Assistant.

The farmer has provided the following information:

Location: {state}
District: {district}
Question: {user_input}

Your task:

1. Use the provided location and district to identify the correct place.
2. Retrieve the latest real-time weather information for that location, including:
   - Temperature
   - Humidity
   - Rainfall (if available)
3. Retrieve the latest soil information for the same location, including:
   - Soil Type
   - Soil Moisture
   - Soil pH
   - Soil Nutrient Status (N, P, K) if available.
4. Analyze both the weather and soil conditions before answering.
5. Answer the farmer's question using the retrieved data.
6. Give only practical and direct advice.
7. Do NOT provide lengthy explanations.
8. If any required data is unavailable, mention it briefly and answer using the available information.
9. Never make up weather or soil data.
10. Respond in the following format:

English:
<Short and direct answer(100 words)>

Hindi:
<Same answer in Hindi>

The response should be clear, concise, and farmer-friendly.
"""
)
prompt = template.invoke({
    "location": state,
    "district": district,
    "question": user_input,

})
model = ChatHuggingFace(llm=llm)
if st.button ('Ask '):
    result = model.invoke(prompt)
    st.write(result.content)
