"""Streamlit app for the ChatGPT clone."""

import dotenv
import langchain
import streamlit as st
import streamlit_chat

dotenv.load_dotenv(dotenv.find_dotenv(), override=True)

st.set_page_config(
    page_title='You Custom Assistant',
    page_icon='🤖'
)
st.subheader('Your Custom ChatGPT 🤖')

chat = langchain.chat_models.ChatOpenAI(
    model_name='gpt-3.5-turbo', temperature=0.5)

# creating the messages (chat history) in the Streamlit session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# creating the sidebar
with st.sidebar:
    # streamlit text input widget for the system message (role)
    system_message = st.text_input(label='System role')
    # streamlit text input widget for the user message
    user_prompt = st.text_input(label='Send a message')

    if system_message:
        if not any(isinstance(x, langchain.schema.SystemMessage) for x in st.session_state.messages):
            st.session_state.messages.append(
                langchain.schema.SystemMessage(content=system_message)
            )

    # if the user entered a question
    if user_prompt:
        st.session_state.messages.append(
            langchain.schema.HumanMessage(content=user_prompt)
        )

        with st.spinner('Working on your request ...'):
            # creating the ChatGPT response
            response = chat(st.session_state.messages)

        # adding the response's content to the session state
        st.session_state.messages.append(
            langchain.schema.AIMessage(content=response.content))


# adding a default SystemMessage if the user didn't entered one
if len(st.session_state.messages) >= 1:
    if not isinstance(st.session_state.messages[0], langchain.schema.SystemMessage):
        st.session_state.messages.insert(0, langchain.schema.SystemMessage(
            content='You are a helpful assistant.'))

# displaying the messages (chat history)
for i, msg in enumerate(st.session_state.messages[1:]):
    if i % 2 == 0:
        streamlit_chat.message(msg.content, is_user=True,
                               key=f'{i} + 🤓')  # user's question
    else:
        streamlit_chat.message(msg.content, is_user=False,
                               key=f'{i} +  🤖')  # ChatGPT response
