from sentence_transformers import SentenceTransformer
import pinecone
from openai import OpenAI
import streamlit as st
client = OpenAI(
    # This is the default and can be omitted
    api_key="",
)
model = SentenceTransformer('all-MiniLM-L6-v2')

pc = pinecone.Pinecone(api_key='')
cloud = 'aws'
region = 'us-east-1'

index_name = 'llama-2-rag-test'

# Check if index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine'
    )
    # Wait for index initialization
    while not pc.describe_index(index_name)['status']['ready']:
        time.sleep(1)

# Connect to Pinecone index
index = pc.Index(index_name)

# def find_match(input):
#     input_em = model.encode(input).tolist()
#     result = index.query(input_em, top_k=2, includeMetadata=True)
#     return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def find_match(input):
    # Encode the input query into a vector
    input_em = model.encode(input).tolist()
    
    # Perform the Pinecone index query using keyword arguments
    result = index.query(
        vector=input_em,  # Pass the vector with a keyword
        top_k=2,          # Number of top matches to return
        include_metadata=True  # Include metadata in the response
    )
    
    # Extract and return the matched texts from the metadata
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):
    response = client.chat.completions.create(
        messages=[
            
            {"role": "system", "content": "You are a helpful assistant that refines queries for better responses from a knowledge base."},
            {"role": "user", "content": f"CONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}
            
        ],
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
        )
    
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string