from tempfile import template
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

import os
HUGGINGFACEHUB_ACCESS_TOKEN = "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"
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN,
    temperature=0.8
)

model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template = 'Generate 5 interesting facts about {topic}',
    input_variables = ['topic']
)
parser = StrOutputParser()

chain = prompt | model | parser

chain.invoke({'topic':'Cricket'})