from setuptools import setup, find_packages

setup(
    name="CAREAI",  # Name of your package
    version="0.1.0",  # Version of your package
    description="A short description of your project",  # Short description
    long_description_content_type="Smart BOM Analyzer",  # Format of long description
    author="Anim Basu",  
    author_email="animbasu1996@outlook.com",  # Author email 
    packages=find_packages(),  # Automatically find packages in your project
    install_requires=[  # List of dependencies
        "numpy",
        "pandas",
        "requests",
        "streamlit",
        "fuzzywuzzy",
        "langchain",
        "langchain_community",
        "pandas",
        "PyPDF2",
        "scikit-learn",
        "dotenv",
        "openai",
        "numpy",
        "plotly",
        "fitz",
        "pymupdf",
        "setuptools"
    ],
    python_requires='>=3.10',  # Minimum Python version required
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
)
