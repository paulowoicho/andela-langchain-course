import os
import time

import dotenv
import utils

data = utils.load_document("files/us_constitution.pdf")
utils.delete_pinecone_index()

index_name = 'us_constitution'
vector_store = utils.insert_or_fetch_embeddings(index_name)
# test = utils.load_from_wikipedia("Satoshi Nakamoto")
# print(test)


if __name__ == '__main__':
    i = 1
    chat_history = []
    print("Write `Quit` or `Exit` to quit")
    while True:
        q = input(f"Question #{i}")
        i = i + 1
        if q.lower() in ["quit", "exit"]:
            print("Qutting")
            time.sleep(2)
            break
        result, _ = utils.ask_with_memory(vector_store, q, chat_history)
        print(result['answer'])
        print("" * 20)
