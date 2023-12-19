import warnings
import time

## Ignoring warning
warnings.filterwarnings("ignore", ".*llm_cache.*")

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import logging
from timeit import Timer

## Get response from model

def getresonse(input_text,no_of_words,style):
    
    ## Calling model
    start_time = time.time()  # Start timer
    llm=CTransformers(model='D:\Must_Research\must_environments\genai-must-1023\Llama_bloke\llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':256})
    
    template= """
              Write a Blog for {style} for the topic {input_text} within {no_of_words} words.
    """

    prompt=PromptTemplate(input_variables=["style","input_text","no_of_words"],
                                           template=template)
    
    response=llm(prompt.format(style=style,input_text=input_text,no_of_words=no_of_words))
    print(response)
    execution_time = time.time() - start_time

    logging.info(f"User input: {input_text}, Style: {style}, Num words: {no_of_words}")
    logging.info(f"Execution time: {execution_time:.2f} seconds")
    return response,execution_time


st.set_page_config(
    page_title="Generate Text",
    page_icon="🦈"
)
st.header("Generate Text 🦈")

input_text=st.text_input("Enter the topic")
no_of_words=st.text_input("how many now of words you need")
style=st.selectbox('Write the content for',
                   ('Researchers','Data Scientists','Common pepole'
                    ,'School'))

submit=st.button("Generate")

if submit:
    res,execution_time=getresonse(input_text,no_of_words,style)
    st.write(res)
    st.write(f"\n**Execution Time:** {execution_time:.2f} seconds")
    



