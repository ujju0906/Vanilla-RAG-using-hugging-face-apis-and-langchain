from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
import requests
from getpass import getpass
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass("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")
# class HuggingFaceAPIModel(LanguageModel):
#     def __init__(self, model_name, api_key, max_new_tokens=256):
#         super().__init__()
#         self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
#         self.headers = {"Authorization": f"Bearer {api_key}"}
#         self.max_new_tokens = max_new_tokens

#     def query(self, prompt):
#         response = requests.post(
#             self.api_url,
#             headers=self.headers,
#             json={"inputs": prompt, "parameters": {"max_new_tokens": self.max_new_tokens}}
#         )
#         return response.json()

#     def generate(self, prompt, **kwargs):
#         full_prompt = "Provide an answer to the following questions: " + prompt
#         output = self.query(full_prompt)
#         return output[0]['generated_text'] 

from openai import OpenAI

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
        
# Initialize the OpenAI client with the API key
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

os.environ["OPENAI_API_KEY"] = "API KEY"

# initializing the embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key='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', model_name="sentence-transformers/all-MiniLM-l6-v2"
)
# default model = "gpt-3.5-turbo"
# llm = HuggingFaceAPIModel(
#     model_name="mistralai/Mistral-7B-Instruct-v0.1",
#     api_key="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"
# )
llm = HuggingFaceHub(huggingfacehub_api_token="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",
    repo_id="huggingfaceh4/zephyr-7b-alpha", 
    model_kwargs={"temperature": 0.5, "max_length": 64,"max_new_tokens":512}
)

directory = "/Users/ujwal_nischal/Desktop/LLM Projects/Vanila Rag/docs"

def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

documents = load_docs(directory)

def split_docs(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

db = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings
)

chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query):
    similar_docs = db.similarity_search(query, k=10) # get 3 closest chunks
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

def mistral7b(user_message, system_message):
    
    # Create a chat completion request
    completion = client.chat.completions.create(
        model="local model",  # Model is currently unused but required for the function call
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
    )
    # Return just the content of the generated message
    return completion.choices[0].message.content  # Adjusted this line
    
system_message = open_file("/Users/ujwal_nischal/Desktop/LLM Projects/Vanila Rag/docs/chatbot1.txt")
    
while True:
    prompt = input(f"{YELLOW}Enter your query here ('exit' to leave): {RESET_COLOR}")
    # Break the loop if a certain condition is met, e.g., if the prompt is 'exit'
    if prompt.lower() == 'exit':
        break

    answer = get_answer(prompt)
    answer2 = f"{CYAN}Context: {answer} \n User Query: {prompt}{RESET_COLOR}"
    print(answer2)

    # chatbot_response = mistral7b(answer2, system_message)
    # print(NEON_GREEN + chatbot_response + RESET_COLOR)
   