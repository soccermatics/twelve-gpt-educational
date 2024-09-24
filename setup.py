from setuptools import setup, find_packages

setup(
    name="twelve-gpt-educational",
    version="0.1",
    packages=find_packages(),  # Finds and includes all your Python packages
    install_requires=[
        "streamlit",
        "pandas",
        "tiktoken",
        # TODO: this is not a complete list
    ],
)
