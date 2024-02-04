import os
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter
import pandas as pd
#from langchain.text_splitter import RecursiveCharacterTextSplitters

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
Batch_size = 10
Chunk_size = 200
textsplitter = NLTKTextSplitter(chunk_size = Chunk_size)
Model = "text-embedding-3-large"
DOCUMENT_PATH = "Hackathon-LLama/360 Lease Document.pdf"

loader = PyPDFLoader(DOCUMENT_PATH)

pages  = loader.load_and_split()
#print(pages[0])

embeddings = []

for start in range(0,len(pages),Batch_size):
    end = start + Batch_size
    batch_pages = [x.page_content for x in pages[start:end]]
    batches = []
    for t in batch_pages:
        batches.extend(textsplitter.split_text(t))
    #print(len(batches))
    #print(batches)
    response = client.embeddings.create(
        model = Model,
        input = batches
    )
    #print(len(response.data[0].embedding)) always the embeddings are 3072 for the latest embedding version -> text-embedding-3-large
    #print(len(batches))
    #print(len(response.data))
    embedding = [response.data[i].embedding for i in range(len(batches))]
    #print(len(embedding))
    #df = pd.DataFrame({"batches":batches,"response.data":embedding})
    embeddings.extend(embedding)

#saving to dataframe
batch = [x.page_content for x in pages]
batches = []
for t in batch:
    batches.extend(textsplitter.split_text(t))
batch = batches
print(len(batch))
print(len(embeddings))
df = pd.DataFrame({"text":batch,"embedding":embeddings})
df.to_csv("embeddings/test.csv")