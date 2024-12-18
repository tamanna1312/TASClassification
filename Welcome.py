import streamlit as st

st.header('Welcome')
st.title("""Hello, welcome to the boron world""")

st.header('Here are four pages from left Main menu', divider='rainbow')


st.subheader(':red[(1) Import data:] upload datafiles from Neptune_Plus and laser')

st.subheader(':green[(2) Data reduction:] choose standard for intra-sequence instrumental correction')

st.subheader(':blue[(3) Data visualization and download:] check results and download files')

st.subheader(':violet[(4) Documention:] Github code and detailed explainations')

st.session_state.all_data = None
