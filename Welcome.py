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


# st.write('An application designed for classification of volcanic rocks using ML.')

st.session_state.all_data = None
