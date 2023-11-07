# -*- coding: utf-8 -*-
import os
import getpass
import numpy as np
import gradio as gr
from langchain.document_loaders import PyMuPDFLoader
def process_uploaded_files(files):
    out=[]
    for file in files:
        if file.split('.')[-1]=='pdf':
            loader = PyMuPDFLoader
        elif file.split('.')[-1]=='txt':
            with open(file, 'r', encoding='utf-8') as f:
                loader = f.read()
        else:
            continue
    return loader

def summary():
    return 0

def qa():
    return 0

with gr.Blocks() as demo:
    gr.Markdown("""<h1><center>AI doc manager</center></h1>""")

    file_input = gr.File(file_types=['text','.pdf'],file_count="multiple", label="Upload docs here(only support .txt and .pdf)")
    submit_button = gr.Button("upload!")


    summary_output = gr.Textbox(placeholder="summary",label="Summary")
    summary_button = gr.Button("get summary!")

    #QA部分
    qa_input = gr.Textbox(placeholder='enter Question here...',label="Question")
    qa_output = gr.Textbox(placeholder='Anwser',label="Answer")
    qa_button = gr.Button("Submit")

    submit_button.click(process_uploaded_files,inputs=file_input, outputs=summary_output)
    # summary_button.click(summary, inputs=text_input, outputs=text_output)
    # qa_button.click(qa, inputs=image_input, outputs=image_output)


if __name__ == "__main__":
    demo.launch()

