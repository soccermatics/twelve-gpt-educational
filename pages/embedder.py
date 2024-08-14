
# Library imports
import streamlit as st
import pandas as pd
import argparse
import tiktoken
import os
from utils.utils import normalize_text

from classes.data_source import PlayerStats
from classes.data_point import Player


from utils.page_components import (
    add_common_page_elements
)

from classes.embeddings import Embeddings



def file_walk(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if not name.endswith('.DS_Store'):  # Skip .DS_Store files
                yield root, name


def get_format(path):
    file_format = "." + path.split(".")[-1]
    if file_format == ".xlsx":
        read_func = pd.read_excel
    elif file_format == ".csv":
        read_func = pd.read_csv
    else:
        raise ValueError(f"File format {file_format} not supported.")
        print("unected file: " + path )
    return file_format, read_func


def embed(file_path,embeddings):
        file_format, read_func = get_format(file_path)
        
        df = read_func(file_path)
        embedding_path = file_path.replace("describe", "embeddings").replace(file_format, ".parquet")

        st.write(embedding_path)

        st.write(df)
        # Check if the content of user exceeds max token length
        tokenizer = tiktoken.get_encoding("cl100k_base")
        df["user_tokens"] = df["user"].apply(lambda x: len(tokenizer.encode(x)))
        df = df[df.user_tokens < 8192]
        df = df.drop("user_tokens", axis=1)

        # Check for common errors in the text
        df["user"] = df["user"].apply(lambda x: normalize_text(x))
        
                    
        df["user_embedded"] = df["user"].apply(
            lambda x: str(embeddings.return_embedding(x))
        )


        
        directory = os.path.dirname(embedding_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        df.to_parquet(embedding_path, index=False)


sidebar_container = add_common_page_elements()

st.divider()

embeddings = Embeddings()

directory= st.text_input("Directory to embedd", "")
st.write("Starting to embedd " + directory)

path_describe = os.path.normpath("data/describe/"+directory)
path_embedded = os.path.normpath("data/embeddings/"+directory)


st.write("Updating all embeddings...")
for root, name in file_walk(path_describe):
    print_path = os.path.join(root, name).replace(path_describe, "")[1:]
    embed(os.path.join(root, name),embeddings)