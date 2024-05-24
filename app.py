import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI


data = pd.read_csv('data/titanic.csv')

st.title("Chat with Titanic Dataset")
st.write("Here is the Titanic dataset:")

with st.expander("Dataframe Preview"):
    st.write(data.head(7))






query = st.text_area("Chat with Dataframe")
st.write(query)

if query:
    llm = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    query_engine = SmartDataframe(data, config={"llm":llm})

    answer = query_engine.chat(query)
    st.write(answer)

