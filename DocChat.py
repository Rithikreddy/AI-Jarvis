import ast #for converting embeddings saved as strings back to arrays
import os
import pandas as pd
import numpy as np
import tiktoken #for counting tokens
from scipy import spatial #for calculating vector similarities for search(cosine similarities)
import torch
import json

from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class DocChat:
    def __init__(
        self,
        EMBEDDING_PATH,
        INTRODUCTION_QUESTION,
        SYSTEM_CONTEXT_MESSAGE,
        DOCUMENT_NAME,
        APIKEY,
        EMBEDDING_MODEL = "text-embedding-3-large",
        GPT_MODEL = "gpt-4-0125-preview",
    ):
        self.EMBEDDING_PATH = EMBEDDING_PATH
        self.INTRODUCTION_QUESTION = INTRODUCTION_QUESTION
        self.SYSTEM_CONTEXT_MESSAGE = SYSTEM_CONTEXT_MESSAGE
        self.DOCUMENT_NAME = DOCUMENT_NAME
        self.EMBEDDING_MODEL = EMBEDDING_MODEL
        self.GPT_MODEL = GPT_MODEL
        df = pd.read_csv(EMBEDDING_PATH)
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
        self.df = df

    #def f(x: float) -> int: return int(x)
    #-> int just tells that f() returns an integer (but it doesn't force the function to return an integer).
    # :float tells people who read the program (and some third-party libraries/programs e. g. pylint) that 
    # x should be a float
    
    def strings_ranked_by_relatednesses(
        self,
        query:str,
        df:pd.DataFrame,
        relatedness_fn = lambda x,y: 1 - spatial.distance.cosine(x,y),
        top_n: int = 300
    ):
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding = []
        query_embedding_response = client.embeddings.create(
            model = self.EMBEDDING_MODEL,
            input = query
        )
        #print("query emb response is ",query_embedding_response.data)
        query_embedding = query_embedding_response.data[0].embedding
        #print(f"query_embedding is {query_embedding}")
        strings_and_relatednesses = [
            (row["text"],relatedness_fn(query_embedding,row["embedding"])) for i,row in df.iterrows()
        ]
        # print(len(strings_and_relatednesses))
        # print("before sorting is ",strings_and_relatednesses[:50])
        strings_and_relatednesses.sort(key = lambda x:x[1],reverse = True)
        # print("len after sorting is ",len(strings_and_relatednesses[:50]))
        # print("After sorting is ",type(strings_and_relatednesses))
        # print("After sorting is ",strings_and_relatednesses[100:110])

        # print("\n\n\n\n string ends here")
        #* operator is powerful for unpacking collections into separate arguments or variables.
        #Finally strings contain all the row["text"] values and relatnesses contains it's corresponding values.
        #zip() function returns a zip object, which is an iterator of tuples -> first item in each passed iterator is paired together
        #second item in each passed iterator are paired together etc.
        strings,relatednesses = zip(*strings_and_relatednesses)
        #print("strings is ",strings[:top_n])
        #print("relatednesses is ",relatednesses[:top_n])
        return strings[:top_n], relatednesses[:top_n]

    def query_message(
        self,
        query:str,
        df:pd.DataFrame,
        model:str,
        token_budget: int
    ) -> str:
        "Returns a message for GPT, with relevant source texts pulled from a dataframe"
        strings,relatednesses = self.strings_ranked_by_relatednesses(query,df)
        intro = f'{self.INTRODUCTION_QUESTION}'
        question = f"\n\nQuestion:{query}"
        message = intro
        for string in strings:
            next_article = f'\n\n {self.DOCUMENT_NAME}\n"""\n{string}\n"""'
            if(
                self.num_tokens(message + next_article + question,model = model) > token_budget
                ):
                    break
            else:
                message += next_article
        return message + question

    def num_tokens(
        self,
        text:str,
        model:str
    ) -> int:
        """Returns the no of tokens in a string"""
        model = self.GPT_MODEL
        #returns the encoding used by a model
        encoding = tiktoken.encoding_for_model(model)
        #encode encodes the text.
        return len(encoding.encode(text))

    def ask(
        self,
        query:str,  #query should be string
        previous_qas,
        token_budget: int = 4096 - 500,
        print_message: bool = False
    ) -> str:
        """Answers a query using GPT and a dataframe of relevant texts and embeddings"""
        model = self.GPT_MODEL
        df = self.df
        message = self.query_message(query,df,model = model,token_budget = token_budget)
        system_message = f"{self.SYSTEM_CONTEXT_MESSAGE}"
        for previous_qa in previous_qas:
            system_message += f"\nYFor this question {previous_qa.get('question')} this is the answer provided {previous_qa.get('answer')}"
        
        #if print_message:
            #print(message)
        messages = [
            {"role":"system","content":system_message},
            {"role":"user","content":message},
        ]
        response = client.chat.completions.create(
            model = model,
            messages = messages,
            temperature = 0
        )
        #print("response is ",response.choices)
        return response.choices[0].message.content

if __name__ == "__main__":
    EMBEDDING_PATH = "embeddings/test.csv"
    INTRODUCTION_QUESTION = 'You are a personal assistant for me. You have inputs from multiple sources. Retrive the answer from one of the source.'
    SYSTEM_CONTEXT_MESSAGE = "Get the relevant answer to the questions from the query asked by the user"
    DOCUMENT_NAME = ""
    API_KEY = os.environ["OPENAI_API_KEY"]
    docChat = DocChat(EMBEDDING_PATH,INTRODUCTION_QUESTION,SYSTEM_CONTEXT_MESSAGE,DOCUMENT_NAME,API_KEY)
    question = "what is Abhinandu Reddy SJSU ID Number?"
    previous_qas = ""
    message = docChat.ask(query=question,previous_qas = previous_qas,print_message = True)
    print("final output is ",message)