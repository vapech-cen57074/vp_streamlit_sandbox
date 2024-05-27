import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data_path = 'data/survey_test_data.csv'
data = pd.read_csv(data_path)

# Exclude specific columns
columns_to_exclude = ["Persona", "Typologie"]
filtered_columns = [col for col in data.columns if col not in columns_to_exclude]

columns_with_placeholder = ['Select column...'] + filtered_columns

st.write(data[filtered_columns].head(7))
selected_column = st.selectbox('Select a column to display:', columns_with_placeholder)

def handle_multiple_choice_values(column_data):
    all_choices = []
    for entry in column_data.dropna():
        choices = entry.split(';')
        all_choices.extend(choices)
    return all_choices

if selected_column != 'Select column...':
    if selected_column in ['Nástroje', 'Činnosti']:
        # Handle multiple choice values
        all_choices = handle_multiple_choice_values(data[selected_column])
        unique_values = list(set(all_choices))
        unique_values_str = ', '.join(unique_values)
        st.write(f"Unique Values of the '{selected_column}' column after separation: {unique_values_str}")

        choices_series = pd.Series(all_choices)
        choices_counts = choices_series.value_counts().sort_values(ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(y=choices_counts.index, x=choices_counts.values, ax=ax)  # Plotting horizontally for better readability
        plt.xlabel('Count')
        plt.ylabel(selected_column)
        plt.title(f'Count of {selected_column} choices')
        st.pyplot(fig)
    else:
        unique_values = data[selected_column].unique()
        unique_values_str = ', '.join(map(str, unique_values))
        st.write(f"Unique Values of the '{selected_column}' column: {unique_values_str}")

        fig, ax = plt.subplots()
        sns.countplot(x=selected_column, data=data, ax=ax, order=data[selected_column].value_counts().index)
        plt.xticks(rotation=45)  # Rotate x labels if needed for better readability
        st.pyplot(fig)
else:
    st.write("No column selected.")
