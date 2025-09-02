import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent import app

st.set_page_config(page_title="🎬 Agentic Movie Discovery", page_icon="🎬")
st.title("🎬 Agentic Movie Discovery System")

# using st.session_state(special dictonary like object which remembers data stored in it across reruns) to not lose our chat history cause every time a user interacts with a widget (like sending a message), Streamlit reruns the entire Python script from top to bottom
if "messages" not in st.session_state: # if it's the first run, we create a new list inside the session state called messages. We start it with a friendly welcome message from the "assistant" to greet the user.
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you find a movie today?"}
    ]

for message in st.session_state.messages:
    # Streamlit command to creates a chat bubble. It looks at the "role" of the message (which will be either "user" or "assistant") and automatically applies the correct styling and avatar.
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) # displaying actual content of the message, st.markdown because it can render formatting like bold text or lists if the AI ever uses them

if prompt := st.chat_input("Ask for a movie recommendation..."): # creating an input box,code inside this block only runs if the user has sent message
    # Adding user message to session state and displaying it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Integration 
    with st.chat_message("assistant"): # a new chat bubble for the assistant's response
        with st.spinner("Thinking..."): # UI feature, displays a small loading animation with the text "Thinking..." while we wait for the agent to process the request and respond.
            langchain_messages = [
                HumanMessage(content=msg["content"]) if msg["role"] == "user" 
                else AIMessage(content=msg["content"]) 
                for msg in st.session_state.messages #  Conversion, We loop through our chat history and convert each message into the official HumanMessage or AIMessage object that our LangGraph agent expects.
            ]
            # We call our compiled agent (app) using the .invoke() method providing it with chat
            response = app.invoke({"messages": langchain_messages})
            
            ai_response_content = response['messages'][-1].content # from agent's final response, from the list of messages grabbing the very last one([-1]), pulling its text content, and saving it.
            
            st.markdown(ai_response_content) # displaying the agent response 
            # most important step for enabling multi-turn conversations, as this updated log will be sent to the agent on the user's next turn, giving it the full context of what has been said so far.
            st.session_state.messages.append({"role": "assistant", "content": ai_response_content}) # saving the agent's latest response to our app" permanent chat log