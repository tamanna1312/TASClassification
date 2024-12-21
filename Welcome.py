import streamlit as st

# st.header('Welcome')
st.header('Welcome')
st.subheader('The application contains 2 sections:')
# st.header(':blue[Using TAS Classifier]')
st.subheader(':blue[Classify Volcanic Rocks using ML model]')
st.write('-> Based on major and minor oxide data.')
st.subheader(':blue[Generate Fake Data from Real-World Data]')
st.write('-> Use real-world data to generate fake data that tries to reflect the geochemical relationships within the oxides.')

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

st.sidebar.image(
    'Goethe-Logo.gif')
# Use columns to arrange the text and image side by side
col1, col2 = st.columns([3, 1])  # Adjust column sizes as needed

with col1:
    # Text content in the first column (left)
    pass

with col2:
    # Add some spacing to move the image further right
    st.markdown("<div style='padding-left: 50px;'></div>", unsafe_allow_html=True)
    # Display the image on the right side with a smaller size
    st.image('Goethe-Logo.gif', width=200)
# st.image('Goethe-Logo.gif', use_column_width=False)
# st.write('An application designed for classification of volcanic rocks using ML.')

st.session_state.all_data = None
