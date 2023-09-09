import os
import pathlib
from typing import List, Optional, Tuple

import dotenv
import langchain
import pinecone

Document = langchain.docstore.document.Document
VectorStore = langchain.vectorstores.base.VST

dotenv.load_dotenv()
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pinecone.init(api_key=pinecone_api_key, environment=pinecone_api_key)

embeddings = langchain.embeddings.huggingface.HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cuda:0'},
    encode_kwargs={'normalize_embeddings': False}
)

llm = langchain.llms.HuggingFacePipeline.from_model_id(
    model_id="bigscience/bloom-1b7",
    task="text-generation",
    device="0",
    model_kwargs={"temperature": 0, "max_length": 64, "do_sample": True},
)


def load_document(file: str) -> Optional[Document]:
    """Given a file path or URL, return the document as a langchain document 
    object.

    Args:
        file (str): The file path or URL to load the document from.

    Returns:
        The document as a langchain document
    """
    print("Loading document...")
    file_extension = pathlib.Path(file).suffix.lower()

    if ".pdf" in file_extension:
        return langchain.document_loaders.PyPDFLoader(file).load()
    elif ".txt" in file_extension:
        return langchain.document_loaders.TextLoader(file).load()
    elif ".docx" in file_extension:
        return langchain.document_loaders.Docx2txtLoader(file).load()
    else:
        print("Document type not supported.")
        return None


def load_from_wikipedia(query: str, lang: str = 'en',
                        load_max_docs: int = 2
                        ) -> Optional[Document]:
    """Given a query, return the document as a langchain document object.

    Args:
        query (str): The query to search Wikipedia for.
        lang (str): The language to search Wikipedia in.
        load_max_docs (int): The maximum number of documents to load.

    Returns:
        The document as a langchain document"""
    return langchain.document_loaders.WikipediaLoader(
        query=query, lang=lang, load_max_docs=load_max_docs).load()


def chunk_data(data: Document,
               chunk_size: int = 256) -> List[Document]:
    """Splits a document into chunks of a given size. A chunk is a smaller
    document.

    Args:
        data (langchain.docstore.document.Document): The document to split.
        chunk_size (int): The size of each chunk.

    Returns:
        A list of documents, each of which is a chunk of the original document.
    """
    text_splitter = langchain.text_splitter.RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0)
    return text_splitter.split_documents(data)


def insert_or_fetch_embeddings(index_name: str,
                               chunks: List[Document]
                               ) -> langchain.vectorstores.Pinecone:
    """Inserts or fetches embeddings from a Pinecone index.

    Args:
        index_name (str): The name of the index to insert or fetch embeddings
            from.
        chunks (List[Document]): A list of documents to insert or fetch
            embeddings from.

    Returns:
        A vector store containing the embeddings.
    """

    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings ... ')
        vector_store = langchain.vectorsrores.Pinecone.from_existing_index(
            index_name, embeddings)
        print('Ok')
    else:
        print(f'Creating index {index_name} and embeddings ...')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vector_store = langchain.vectorsrores.Pinecone.from_documents(
            chunks, embeddings, index_name=index_name)
        print('Ok')

    return vector_store


def delete_pinecone_index(index_name: str = 'all') -> None:
    """Deletes a Pinecone index.

    Args:
        index_name (str): The name of the index to delete. If 'all', all
            indexes are deleted.
    """

    if index_name == 'all':
        indexes = pinecone.list_indexes()
        print('Deleting all indexes ... ')
        for index in indexes:
            pinecone.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pinecone.delete_index(index_name)
        print('Ok')


def ask_and_get_answer(vector_store: langchain.vectorstores.Pinecone,
                       query: str) -> str:
    """Given a vector store and a query, return the answer to the query.

    Args:
        vector_store (langchain.vectorstores.Pinecone): The vector store to
            search for the answer.
        query (str): The query to search for the answer to.

    Returns:
        The answer to the query."""

    retriever = vector_store.as_retriever(
        search_type='similarity', search_kwargs={'k': 3})

    chain = langchain.chains.RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever)

    return chain.run(query)


def ask_with_memory(vector_store: langchain.vectorstores.Pinecone,
                    question: str, chat_history: List = []) -> Tuple:
    """Given a vector store, a question, and a chat history, return the answer.

    Args:
        vector_store (langchain.vectorstores.Pinecone): The vector store to
            search for the answer.
        question (str): The question to search for the answer to.
        chat_history (list): A list of tuples containing the question and
            answer pairs.

    Returns:
        A tuple containing the answer and the updated chat history."""

    retriever = vector_store.as_retriever(search_type='similarity',
                                          search_kwargs={'k': 3})

    crc = langchain.chains.ConversationalRetrievalChain.from_llm(
        llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))

    return result, chat_history