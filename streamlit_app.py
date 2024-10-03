import streamlit as st

def main():
# Title and description of the homepage
    st.title("Smart BOM Analyzer")
    st.write("Welcome to the Smart BOM Analyzer. Analyze your Bill of Materials easily.")


    st.page_link("bom_analyzer.py", label="Click Here to get Started", icon="🌎")

if __name__ == '__main__':
    main()