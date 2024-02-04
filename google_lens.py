import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='vision_key.json'
from google.cloud import vision
from openai import OpenAI
import pandas as pd
from langchain.text_splitter import NLTKTextSplitter
import base64
import requests

Chunk_size = 200

textsplitter = NLTKTextSplitter(chunk_size = Chunk_size)

api_key = "<keep your api key>"

vision_client = vision.ImageAnnotatorClient()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

Model = "text-embedding-3-large"

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Getting the base64 string
image_path = 'praneeth.jpeg'
base64_image = encode_image(image_path)

with open(image_path, 'rb') as image_file:
    content = image_file.read()
    image = vision.Image(content = content)
    # IMAGE_URI = 'id.jpeg'
    # image.source.image_uri = IMAGE_URI
    response = vision_client.text_detection(image=image)
    text = response.text_annotations[0].description
    text_with_commas = text.replace('\n', ',')
    text = text_with_commas
    #print(text)
    # response = client.embeddings.create(
    #     model = Model,
    #     input = text
    # )
    # embeddings = [response.data[0].embedding]
    # csv_file_path = "embeddings/360_Lease_Document.csv"

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4-vision-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is inside the image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
response_dict = response.json()  # Parse the JSON response to a Python dictionary

# Extract the 'content' attribute from the first choice of the response
content = response_dict['choices'][0]['message']['content']
#print(content)  # Prints the content of the message

text += "\n"+content
print("combined text is ",text)
response = client.embeddings.create(
    model = Model,
    input = text
)

embeddings = [response.data[0].embedding]
csv_file_path = "embeddings/test.csv"
df = pd.DataFrame({"text":text,"embedding":embeddings})
#df.to_csv("embeddings/test.csv")
#df = pd.read_csv(csv_file_path)
# df = pd.DataFrame({"text":text,"embedding": embeddings})       #follow the same spellings
# Save updated DataFrame back to CSV
df.to_csv(csv_file_path, mode='a', header=False)