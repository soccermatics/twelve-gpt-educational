import streamlit as st
import pandas as pd
import openai
from utils.embeddings_utils import get_embedding, cosine_similarity
import os

from settings import (
    ENGINE_ADA,
    GPT3_BASE,
    GPT3_VERSION,
    GPT3_KEY,
    USE_GEMINI,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_API_KEY,
)


class Embeddings:
    def __init__(self):
        self.df_dict = None

    def search(self, query, top_n=3):
        # type is the index into the various dataframes stored in the embeddings.
        # if type is not specified, it will search all dataframes
        # otherwise it will search those listed

        if USE_GEMINI:
            import google.generativeai as genai

            genai.configure(api_key=GEMINI_API_KEY)
            ENGINE = GEMINI_EMBEDDING_MODEL
        else:
            openai.api_base = GPT3_BASE
            openai.api_version = GPT3_VERSION
            openai.api_key = GPT3_KEY
            # text-embedding-ada-002 (Version 2) model
            ENGINE = ENGINE_ADA
        embedding = get_embedding(query, engine=ENGINE, use_gemini=USE_GEMINI)

        # An option for the future is to take the top from each dataframe, so we get a mixture of responses.
        df = self.df_dict
        df["similarities"] = df.user_embedded.apply(
            lambda x: cosine_similarity(x, embedding)
        )
        df = df[df.similarities > 0.7]

        res = df.sort_values("similarities", ascending=False).head(top_n)
        return res

    def compare_strings(self, string1, string2):
        # Use this function to compare two strings or words embeddings
        # Returns co-sine similarilty between the two strings
        ENGINE = GEMINI_EMBEDDING_MODEL if USE_GEMINI else ENGINE_ADA
        embedding1 = get_embedding(string1, engine=ENGINE, use_gemini=USE_GEMINI)
        embedding2 = get_embedding(string2, engine=ENGINE, use_gemini=USE_GEMINI)

        return cosine_similarity(embedding1, embedding2)

    def return_embedding(self, query):
        if USE_GEMINI:
            import google.generativeai as genai

            genai.configure(api_key=GEMINI_API_KEY)
            ENGINE = GEMINI_EMBEDDING_MODEL
        else:
            openai.api_base = GPT3_BASE
            openai.api_version = GPT3_VERSION
            openai.api_key = GPT3_KEY
            # text-embedding-ada-002 (Version 2) model
            ENGINE = ENGINE_ADA
        embedding = get_embedding(query, engine=ENGINE, use_gemini=USE_GEMINI)

        return embedding


class PlayerEmbeddings(Embeddings):
    def __init__(self):
        self.df_dict = PlayerEmbeddings.get_embeddings()

    def get_embeddings():
        # Gets all embeddings
        df_embeddings_dict = dict()

        files = [
            "Interpretation",
            "Forward",
        ]

        df_embeddings = pd.DataFrame()
        for file in files:
            # Read in
            df_temp = pd.read_parquet(f"data/embeddings/{file}.parquet")
            if "category" not in df_temp:
                df_temp["category"] = None
            if "format" not in df_temp:
                df_temp["format"] = None
            df_temp = df_temp[
                ["user", "assistant", "category", "user_embedded", "format"]
            ]
            df_temp["user_embedded"] = df_temp.user_embedded.apply(eval).to_list()
            df_embeddings = pd.concat([df_embeddings, df_temp], ignore_index=True)

        return df_embeddings


class CountryEmbeddings(Embeddings):
    def __init__(self):
        self.df_dict = CountryEmbeddings.get_embeddings()

    def get_embeddings():
        # Gets all embeddings
        df_embeddings_dict = dict()

        files = [
            "WVS_qualities",
        ]

        df_embeddings = pd.DataFrame()
        for file in files:
            # Read in
            df_temp = pd.read_parquet(f"data/embeddings/{file}.parquet")
            if "category" not in df_temp:
                df_temp["category"] = None
            if "format" not in df_temp:
                df_temp["format"] = None
            df_temp = df_temp[
                ["user", "assistant", "category", "user_embedded", "format"]
            ]
            df_temp["user_embedded"] = df_temp.user_embedded.apply(eval).to_list()
            df_embeddings = pd.concat([df_embeddings, df_temp], ignore_index=True)

        return df_embeddings


class PersonEmbeddings(Embeddings):
    def __init__(self):
        self.df_dict = PersonEmbeddings.get_embeddings()

    def get_embeddings():
        # Gets all embeddings
        df_embeddings_dict = dict()

        files = [
            "Forward_bigfive",
        ]

        df_embeddings = pd.DataFrame()
        for file in files:
            # Read in
            df_temp = pd.read_parquet(f"data/embeddings/{file}.parquet")
            if "category" not in df_temp:
                df_temp["category"] = None
            if "format" not in df_temp:
                df_temp["format"] = None
            df_temp = df_temp[
                ["user", "assistant", "category", "user_embedded", "format"]
            ]
            df_temp["user_embedded"] = df_temp.user_embedded.apply(eval).to_list()
            df_embeddings = pd.concat([df_embeddings, df_temp], ignore_index=True)

        return df_embeddings
