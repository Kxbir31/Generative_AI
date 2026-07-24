from tempfile import template
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv



load_dotenv()

import os
HUGGINGFACEHUB_ACCESS_TOKEN = "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"
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN ,
    temperature=0.8
)
st.header("HealthBot")
symptoms = st.text_input("Enter Your Symptoms")
time = st.text_input("From how long you are suffering from the Symptoms")
age = st.text_input("Enter Your Age")
gender = st.text_input("Enter Your Gender")

template = PromptTemplate(
    input_variables=["symptoms", "time", "age", "gender"],
    template=f"""
You are an experienced AI Healthcare Assistant.

Patient Details:
- Symptoms: {symptoms}
- Duration of Symptoms: {time}
- Age: {age}
- Gender: {gender}

Your task:

1. Analyze the symptoms along with the patient's age, gender, and duration of illness.
2. Identify the most likely disease(s) or health condition(s). Do not claim certainty.
3. Mention only the 1–3 most probable conditions.
4. Suggest simple home-care or first-aid measures that may help relieve the symptoms until the patient consults a healthcare professional.
5. If the symptoms indicate a medical emergency, clearly advise the patient to seek immediate medical attention.
6. Do NOT prescribe prescription medicines or provide dosage instructions.
7. Keep the response short, practical, and easy to understand.
8. Do not provide unnecessary explanations.
9. Include a disclaimer that this is not a medical diagnosis.

Respond in the following format:

Possible Condition(s):
• <Condition 1>
• <Condition 2> (if applicable)
• <Condition 3> (if applicable)

Immediate Relief:
• <Short actionable advice>

When to See a Doctor:
• <One-line recommendation>

Give me a short answer in hindi also
Disclaimer:
This is only an AI-generated health suggestion and not a medical diagnosis. Please consult a qualified healthcare professional for an accurate diagnosis and treatment.
"""
)

prompt = template.invoke({
    "Symptoms":symptoms,
    "Time":time,
    "Age":age,
    "Gender":gender,

})

model = ChatHuggingFace(llm = llm)

if st.button("Analyze"):
    result = model.invoke(prompt)
    st.write(result.content)

