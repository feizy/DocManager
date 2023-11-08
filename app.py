# -*- coding: utf-8 -*-
import os
import getpass
import numpy as np
import gradio as gr
from langchain.document_loaders import PyMuPDFLoader
from langchain.llms import AzureOpenAI
import os
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


def process_uploaded_files(files):
    global texts,docsearch
    contents = ""
    for file in files:
        if file.split('.')[-1] == 'pdf':
            loader = PyMuPDFLoader(file)
            data = loader.load()
            for f in data:
                contents += f.page_content
        elif file.split('.')[-1] == 'txt':
            with open(file, 'r', encoding='utf-8') as f:
                loader = f.read()
            contents += loader
        else:
            continue
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(contents)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))])
    return 0

def summary(language):
    # contents=""
    # for file in files:
    #     if file.split('.')[-1]=='pdf':
    #         loader = PyMuPDFLoader(file)
    #         data=loader.load()
    #         for f in data:
    #             contents+= f.page_content
    #     elif file.split('.')[-1]=='txt':
    #         with open(file, 'r', encoding='utf-8') as f:
    #             loader = f.read()
    #         contents += loader
    #     else:
    #         continue
    # text_splitter = CharacterTextSplitter()
    # texts = text_splitter.split_text(contents)
    docs = [Document(page_content=t) for t in texts[:3]]
    prompt_template = """Write a concise summary of the following:

    {text}

    CONCISE SUMMARY IN {language}:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text", "language"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    with get_openai_callback() as cost:
        output=chain.run({'input_documents':docs,'language':language})
    return output, cost

def qa(query,language):
    # embeddings = OpenAIEmbeddings()
    # docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))])
    docs = docsearch.similarity_search(query)
    chain = load_qa_with_sources_chain(llm, chain_type="stuff")
    query = f"Answer the question in {language}"
    with get_openai_callback() as cost:
        output = chain.run({"input_documents": docs, "question": query})
    return output, cost

with gr.Blocks() as demo:
    gr.Markdown("""<h1><center>AI doc manager</center></h1>""")
    file_input = gr.Files(file_types=['text','.pdf'], label="Upload your document(only support .txt and .pdf)")
    submit_button = gr.Button("upload!")
    language_input = gr.Textbox(placeholder="Input your desired output language", label="Language")
    summary_output = gr.Textbox(placeholder="summary", label="Summary")
    cost_output = gr.Textbox(placeholder="AI is not free!", label="Cost")
    summary_button = gr.Button("Get Summary!")

    #QA部分
    qa_input = gr.Textbox(placeholder='enter Question here...',label="Question")
    qa_output = gr.Textbox(placeholder='Anwser',label="Answer")
    qa_button = gr.Button("Question Submit")

    submit_button.click(process_uploaded_files,inputs=file_input)
    summary_button.click(summary, inputs=[language_input], outputs=[summary_output, cost_output])
    qa_button.click(qa, inputs=[qa_input, language_input], outputs=[qa_output, cost_output])


if __name__ == "__main__":
    _ = load_dotenv(find_dotenv())
    os.environ["OPENAI_API_TYPE"] = os.getenv("OPENAI_API_TYPE")
    os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
    os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    llm = AzureOpenAI(
        deployment_name="xiwe-test-davinci-003",
        model_name="text-davinci-003",
    )
    demo.launch(debug=True)

