import streamlit as st


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url('https://raw.githubusercontent.com/tamanna1312/TASClassification/main/Image2.gif');
                background-repeat: no-repeat;
                background-size: 200px; /* Adjust the size as needed */
                background-position: center top; /* Center the image */
                padding-top: 200px; /* Add padding below the image */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()



st.header('Welcome')
st.subheader('The application contains 2 sections:')
# st.header(':blue[Using TAS Classifier]')
st.subheader(':blue[Classify Volcanic Rocks using ML model]')
st.write('-> Based on major and minor oxides.')
st.subheader(':blue[Generate Fake Data from Real-World Data]')
st.write('-> Use real-world data to generate fake data that tries to reflect the geochemical relationships within the oxides.')


col1, col2 = st.columns([3, 1])  

with col1:
    pass

with col2:
    # Add some spacing to move the image further right
    st.markdown("<div style='padding-left: 50px;'></div>", unsafe_allow_html=True)
    # Display the image on the right side with a smaller size
    st.image('Goethe-Logo.gif', width=200)
# st.image('Goethe-Logo.gif', use_column_width=False)
# st.write('An application designed for classification of volcanic rocks using ML.')

st.session_state.all_data = None
