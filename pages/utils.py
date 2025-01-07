# utils.py

import streamlit as st

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url('https://raw.githubusercontent.com/tamanna1312/TASClassification/main/Applogo.jpg');
                background-repeat: no-repeat;
                background-size: 150px 150px; /* Explicit width and height */
                background-position: 30px 10px; /* Adjust position */
                padding-top: 170px; /* Add space below to separate from text */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def add_sidebar_image():
    st.sidebar.image('Goethe-Logo.gif', use_column_width=True)
