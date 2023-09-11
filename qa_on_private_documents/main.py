"""Main script to run the QA system on private documents"""
import time

import utils

data = utils.load_document("files/us_constitution.pdf")
utils.delete_pinecone_index()

INDEX_NAME = 'us-constitution'

chunks = utils.chunk_data(data)
vector_store = utils.insert_or_fetch_embeddings(INDEX_NAME, chunks)


if __name__ == '__main__':
    i = 1
    chat_history = []
    print("Write `Quit` or `Exit` to quit")
    while True:
        q = input(f"Question #{i}: ")
        i = i + 1
        if q.lower() in ["quit", "exit"]:
            print("Quitting")
            time.sleep(2)
            break
        result, _ = utils.ask_with_memory(vector_store, q, chat_history)
        print(result['answer'])
        print("" * 20)
