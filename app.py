# -*- coding: utf-8 -*-
import numpy as np
import gradio as gr
import langchain
def process_uploaded_files(files):
    file_contents = []
    for file in files:
        if file[]

def summary():
    return 0

def qa():
    return 0

with gr.Blocks() as demo:
    gr.Markdown("""<h1><center>AI doc manager</center></h1>""")

    file_input = gr.File(file_types=['.txt','.pdf'],file_count="multiple")
    submit_button = gr.Button("upload!")


    summary_output = gr.Textbox(placeholder="summary",label="Summary")
    summary_button = gr.Button("get summary!")

    #QA部分
    qa_input = gr.Textbox(placeholder='enter Question here...',label="Question")
    qa_output = gr.Textbox(placeholder='Anwser',label="Answer")
    qa_button = gr.Button("Submit")


    # summary_button.click(summary, inputs=text_input, outputs=text_output)
    # qa_button.click(qa, inputs=image_input, outputs=image_output)


if __name__ == "__main__":
    demo.launch()