## Langchain Projects

This repo contains code for projects written for the [Langchain course](https://www.udemy.com/course/master-langchain-pinecone-openai-build-llm-applications/)
provided by Andela.

### Projects

- [QA on Private Documents](qa_on_private_documents): Demonstrates how one might use Langchain for QA
    on private documents. The code is in the `qa_on_private_documents` directory.


### Running the code

To run create a virtual environment and install the requirements:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install the requirements:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project directory and add the following:

```bash
PINECONE_API_KEY="your pinecone api key"
HUGGINGFACE_TOKEN="your huggingface token"
PINECONE_ENVIRONMENT="your pinecone environment name"
```

Then run the main.py file in the project directory:

```bash
python3 main.py
```