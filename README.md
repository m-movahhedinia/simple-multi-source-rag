# Simple multi-source RAG
This is a very rudimentary rag implementation with multi-source enrichment. The code is meant to be just a personal 
playground. Extensive testing is needed before using anywhere.

## Run the code
First, install the requirements.
```shell
pip install -r requirements.txt
```
Second, add your huggingface access token and pinecone credentials to the code, line 26 to 28. You can use the links 
below to get them.<br>
https://huggingface.co/docs/hub/security-tokens<br>
https://docs.pinecone.io/docs/quickstart

Then run the command bellow from the project root:
```shell
streamlit run main.py
```

The web UI should load in your browser. It may take it a few seconds to load completely.
You can use the sidebar to upload one or more PDFs.
The instructions get appended to the default prompt.
For better performance, change the code to use one of OpenAI's models.

