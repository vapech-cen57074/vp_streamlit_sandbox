import streamlit as st
import pandas as pd
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

@st.cache_data
def load_data():
    data = pd.read_csv('data/titanic.csv')
    return data

def get_insights(question, data):
    # Define your prompt template
    template = """
    The dataset contains information about passengers on the Titanic. 
    The columns are: {columns}.
    Here is the first few rows of the dataset: {data_head}
    
    Question: {question}
    Answer:
    """
    
    # Prepare the input for the prompt
    columns = ", ".join(data.columns)
    data_head = data.head().to_dict(orient='records')
    input_data = {
        "columns": columns,
        "data_head": data_head,
        "question": question
    }
    
    prompt = PromptTemplate(template=template, input_variables=["columns", "data_head", "question"])

    # Generate response using OpenAI
    llm = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    formatted_prompt = prompt.format(**input_data)
    
    # Create a RunnableLambda with the llm callable
    runnable = RunnableLambda(lambda x: llm(x))
    
    # Execute the runnable using the invoke method
    result = runnable.invoke(formatted_prompt)
    
    return result

# Streamlit App
def main():
    st.title("Chat with Titanic Dataset")
    
    # Load data
    data = load_data()
    
    # Display data
    st.write("Here is the Titanic dataset:")
    st.write(data)
    
    # Input for user's question
    question = st.text_input("Ask a question about the dataset:")
    
    if question:
        st.write("You asked:", question)
        
        # Get insights
        insights = get_insights(question, data)
        
        # Display insights
        st.write("Insight:", insights)

if __name__ == "__main__":
    main()
