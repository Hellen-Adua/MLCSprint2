import streamlit as st

# Prompts for each section
section_prompts = [
    "Introduction",
    "Import Modules",
    "Load the Data",
    "Exploratory Data Analysis",
    "Visualize with Pairplots",
    "Data Visualisation",
    "Data Preprocessing and Feature Engineering",
    "Model Training ",
    "Training Model with PCA reduced data",
    "Model Evaluation",
    " Comparison of metrics",
    "Summary"
  
]

# Sidebar for section selection
section = st.sidebar.selectbox("Choose a section", section_prompts)