from tempfile import template
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
st.header('Research_Tool')
user_input = st.text_input('Enter your Prompt')
import os

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    temperature=0.8
)
model = ChatHuggingFace(llm = llm)
paper_input = st.selectbox("Select Research paper name ", ["Normal","Creative","Proffesional"])

style_input = st.selectbox("Select the Explanation Style ",["Beginner-Friendly","Technical","Code Oriented","Maths oriented"])

Length_input = st.selectbox("Enter your explanation length",["short (1-2 Para)","Medium(3-4 para)","long (5-8 para)"])

template = PromptTemplate(
    input_variables=[
        "user_input",
        "paper_input",
        "style_input",
        "length_input"
    ],
    template=f"""
You are an expert research assistant.

Your task is to create a concise and well-structured research summary based on the user's requirements.

User Inputs:
- Topic:{user_input}
- Title Type: {paper_input}
- Writing Style: {style_input}
- Summary Length: {Length_input}

Instructions:

1. Generate an appropriate title according to the requested title type.
2. Write a research-based summary on the given topic.
3. Use the requested writing style throughout the response.
4. Ensure the summary matches the requested length.
5. Focus on the most important concepts, findings, applications, advantages, limitations, and future scope (where applicable).
6. Present information in a logical flow with clear headings.
7. Do not include unnecessary filler or unrelated information.
8. The information should be accurate, easy to understand, and suitable for academic use.

Output Format:

# <Generated Title>

## Overview
...

## Key Concepts
...

## Important Findings
...

## Applications
...

## Advantages
...

## Limitations
...

## Future Scope
...

## Conclusion
...
"""
)
prompt = template.invoke({
    "Title_input": user_input,
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": Length_input,
})


if st.button ('Summarise'):
    result = model.invoke(prompt)
    st.write(result.content)