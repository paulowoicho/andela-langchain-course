"""Main script to run the QA system on private documents"""
import os

import streamlit as st
import utils


if __name__ == '__main__':
    st.image('files/img.png')
    st.subheader('LLM Question-Answering Application 🤖')
    with st.sidebar:
        # file uploader widget
        uploaded_file = st.file_uploader(
            'Upload a file:', type=['pdf', 'docx', 'txt'])
        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100,
                                     max_value=2048, value=512,
                                     on_change=utils.clear_history)
        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3,
                            on_change=utils.clear_history)

        index_name = st.text_input('Index name:', value='us-constitution')

        # add data button widget
        add_data = st.button('Add Data', on_click=utils.clear_history)

        if uploaded_file and add_data:  # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = utils.load_document(file_name)
                chunks = utils.chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                # creating the embeddings and returning the Chroma vector store
                utils.delete_pinecone_index()
                vector_store = utils.insert_or_fetch_embeddings(index_name, 
                                                                chunks)

                # saving the vector store in the streamlit session state 
                # (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')
        
        # user's question text input widget
        q = st.text_input('Ask a question about the content of your file:')
        if q:  # if the user asked a question
            standard_answer = ("Answer only based on the text you received as "
                               "input. Don't search external sources. If you "
                               "can't answer then return `I DONT KNOW`.")
            q = f"{q} {standard_answer}"
            if 'vs' in st.session_state:
                vector_store = st.session_state.vs
                st.write(f'k: {k}')
                answer = utils.ask_and_get_answer(vector_store, q, k)

                # text area widget for the LLM answer
                st.text_area('LLM Answer: ', value=answer)

                st.divider()

                # if there's no chat history in the session state, create it
                if 'history' not in st.session_state:
                    st.session_state.history = ''

                # the current question and answer
                value = f'Q: {q} \nA: {answer}'

                st.session_state.history = (
                    f'{value} \n {"-" * 100} \n {st.session_state.history}')
                h = st.session_state.history

                # text area widget for the chat history
                st.text_area(label='Chat History', value=h, key='history',
                             height=400)
