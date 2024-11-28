from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *  # Make sure 'utils' contains the necessary functions like `get_conversation_string`, `query_refiner`, and `find_match`

st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

# Initialize session state for responses and requests
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Initialize the language model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="")

# Initialize memory for the conversation
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Define system and human message templates
system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Answer the question as truthfully as possible using the provided context, 
    and if the answer is not contained within the text below, say 'I don't know'"""
)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

# Create a chat prompt template
prompt_template = ChatPromptTemplate.from_messages([
    system_msg_template,
    MessagesPlaceholder(variable_name="history"),
    human_msg_template
])

# Initialize conversation chain
conversation = ConversationChain(
    memory=st.session_state.buffer_memory,
    prompt=prompt_template,
    llm=llm,
    verbose=True
)

# Initialize session state for query and history
if 'query' not in st.session_state:
    st.session_state["query"] = ""
if 'requests' not in st.session_state:
    st.session_state.requests = []
if 'responses' not in st.session_state:
    st.session_state.responses = []

# Container for chat history
response_container = st.container()
# Container for text box
textcontainer = st.container()

with textcontainer:
    # Display the text input field
    query = st.text_input(
        "Query:",
        value=st.session_state.query,
        max_chars=None,
        key="query",
        placeholder="Type the Query",
        disabled=False,
        label_visibility="visible"
    )

    # Button to submit the query
    if st.button("Submit"):
        if query:
            with st.spinner("Generating..."):
                conversation_string = get_conversation_string()
                # st.code(conversation_string)
                # refined_query = query_refiner(conversation_string, query)
                # st.subheader("Refined Query:")
                # st.write(refined_query)
                context = find_match(query)
                # print(context)  
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")

            # Save query and response to session state
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)
            
        
        
        
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

          