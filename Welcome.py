import streamlit as st

st.set_page_config(
    page_title="Welcome!",
    page_icon="👋",
)

# Use markdown to set the title in the sidebar
st.sidebar.markdown("# Welcome! 👋")

st.write("# Welcome to my demo app! 👋")
st.write(
    "In this demo app, you can explore some features of Streamlit.\n\n"
    "In the Survey Analysis section, you can create a countplot that shows the distribution for a selected column.\n\n"
    "In the Titanic Prediction section, you can predict the survival of a generated passenger based on your inputs."
)


