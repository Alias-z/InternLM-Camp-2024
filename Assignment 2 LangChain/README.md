# Assignment 2: Personal Knowledge DataBase with InternLM + LangChain
**Tutorial doc**: https://github.com/InternLM/tutorial/blob/main/langchain/readme.md
<br>
**Tutorial video**: https://www.bilibili.com/video/BV1sT4y1p71V/?share_source=copy_web&vd_source=7ef00cba9894eae747b373127e0807d4
<br>
**Tasks**:
1. Reproduce the pipeline
2. Deploy an expert database via OpenXLab on any given field (Here I choose MMDetection and MMEngine)


### InternLM deployment (similar to assignment 1)
1. Start a A100 (1/4) runtime with `Cuda11.7-conda` docker image on [InternStudio](https://studio.intern-ai.org.cn/)

2. Create a new environment in the terminal (bash):
`conda create --name langchain --clone=/root/share/conda_envs/internlm-base`

3. After activating the new environment `conda activate langchain`, we need to install the dependencies of InternLM and LangChain:
```
python -m pip install --upgrade pip
pip install modelscope==1.9.5 transformers==4.35.2 streamlit==1.24.0 sentencepiece==0.1.99 accelerate==0.24.1  # for InternLM
pip install cmake lit langchain==0.0.292 gradio==4.4.0 chromadb==0.4.15 sentence-transformers==2.2.2 unstructured==0.10.30 markdown==3.3.7  # for LangChain
pip install -U huggingface_hub  # to download Sentence-Transformer weights
```

4. Copy and paste the InternLM-7B model weights from the `/share` folder
```
mkdir -p /root/data/model/Shanghai_AI_Laboratory
cp -r /root/share/temp/model_repos/internlm-chat-7b /root/data/model/Shanghai_AI_Laboratory
```

5. Download Sentence-Transformer weights via huggingface_hub
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir /root/data/model/sentence-transformer')
```

6. Download NLTK data and unzip them
```
cd /root
git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages
cd nltk_data
mv packages/*  ./
cd tokenizers
unzip punkt.zip
cd ../taggers
unzip averaged_perceptron_tagger.zip
```

### Build a Database regarding InternLM docs

Use the docs (markdown and txt) from the following repositories as text data:
[OpenCompass](https://gitee.com/open-compass/opencompass), [IMDeploy](https://gitee.com/InternLM/lmdeploy), [XTuner](https://gitee.com/InternLM/xtuner), [InternLM-XComposer](https://gitee.com/InternLM/InternLM-XComposer), [Lagent](https://gitee.com/InternLM/lagent), and [InternLM](https://gitee.com/InternLM/InternLM)

To test task 2,  I also cloned [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMEngine](https://github.com/open-mmlab/mmengine) as **database 2**.

1. git clone code repositories
```
mkdir -p /root/data
cd /root/data
git clone https://github.com/InternLM/tutorial  # code
git clone https://gitee.com/open-compass/opencompass.git
git clone https://gitee.com/InternLM/lmdeploy.git
git clone https://gitee.com/InternLM/xtuner.git
git clone https://gitee.com/InternLM/InternLM-XComposer.git
git clone https://gitee.com/InternLM/lagent.git
git clone https://gitee.com/InternLM/InternLM.git
git clone https://github.com/open-mmlab/mmdetection.git
git clone https://github.com/open-mmlab/mmengine.git
```

2. Filter out docs that are not markdown nor txt
```python
import os 
def get_files(dir_path: str) -> list:
    """Filter out docs that are not markdown nor txt"""
    file_list = [] # to store the paths of .md and .txt files
    for filepath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith(".md"):
                file_list.append(os.path.join(filepath, filename))  # add .md file paths
            elif filename.endswith(".txt"):
                file_list.append(os.path.join(filepath, filename))  # add .txt file paths
    return file_list
```

3. Convert .md and .txt files to plain text and load them
```python
from tqdm import tqdm
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader

def get_text(target_dirs: list) -> list:
    """Convert .md and .txt files to plain text and load them under given target directories"""
    docs = []  # to store unformated text
    for dir_path in target_dirs:
        file_lst = get_files(dir_path)  # get the paths of .md and .txt
        for one_file in tqdm(file_lst):
            file_type = one_file.split('.')[-1]
            if file_type == 'md':
                loader = UnstructuredMarkdownLoader(one_file)  # convert .md to plain text
            elif file_type == 'txt':
                loader = UnstructuredFileLoader(one_file)  # convert .txt to plain text
            else:
                continue
            docs.extend(loader.load())  # add the converted plain text to a list
    return docs

target_dirs = [
    "/root/data/InternLM",
    "/root/data/InternLM-XComposer",
    "/root/data/lagent",
    "/root/data/lmdeploy",
    "/root/data/opencompass",
    "/root/data/xtuner"
]

target_dirs_2 = [
    "/root/data/mmdetection",
    "/root/data/mmengine",
]

docs = get_text(target_dirs)
docs_2 = get_text(target_dirs_2)
```

4. Split the docs into chunks, and vectorize the chunks to build the database
```python
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")  # model for chuncks vectorization

persist_directory = '/root/data_base/vector_db/chroma'  # where to store the vector database
split_docs = text_splitter.split_documents(docs)  # splitted text chuncks

persist_directory_2 = '/root/data_base/vector_db/chroma_2'  # where to store the vector database
split_docs_2 = text_splitter.split_documents(docs_2)  # splitted text chuncks

vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory
)

vectordb_2 = Chroma.from_documents(
    documents=split_docs_2,
    embedding=embeddings,
    persist_directory=persist_directory_2
)

vectordb.persist()  # store the database in the local disk
vectordb_2.persist()
```

### Chain InternLM to LangChain
1. Define InternLM as a LLM class of LangChain and store it as `LLM.py`
```Python
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class InternLM_LLM(LLM):
    """Define InternLM as a LLM class in LangChain"""
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path :str):
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
        self.model = self.model.eval()
        print("完成本地模型的加载")

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
        - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
        - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
        """
        
        messages = [(system_prompt, '')]
        response, history = self.model.chat(self.tokenizer, prompt , history=messages)
        return response
        
    @property
    def _llm_type(self) -> str:
        return "InternLM"  # tag name
```

2. Realize the RAG pipeline with LangCahin

First, We need to define the QA chain
```Python
import os
from LLM import InternLM_LLM
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

def load_chain():
    """Custom QA chain"""
    embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")  # embeddings

    persist_directory = '/root/data_base/vector_db/chroma' # database directory

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)  # get vector database


    llm = InternLM_LLM(model_path = "/root/data/model/Shanghai_AI_Laboratory/internlm-chat-7b")  # custom LLM

    # Prompt Template
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)  # context and question would be similar text chunck and user prompt in practice

    qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})  # realize QA using LangChain   
    return qa_chain
```
We would also need a key function allowing interactive responses in a web demo
```Python
class Model_center():
    """To store the QA chian objects"""
    def __init__(self):
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """ENTER questions to get answers"""
        if question == None or len(question) < 1:
            return "", chat_history  # if no question asked
        try:
            chat_history.append((question, self.chain({"query": question})["result"]))  # add previous answers to the chat history
            return "", chat_history
        except Exception as e:
            return e, chat_history
```
Finally, we need code to create a gradio demo
```Python
import gradio as gr

model_center = Model_center()

block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            gr.Markdown("""<h1><center>InternLM</center></h1>
                <center>书生浦语</center>
                """)  # web title

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=450, show_copy_button=True)  # create a chatbot
            msg = gr.Textbox(label="Prompt/问题")  # creat a TextBox for entering questions

            with gr.Row():
                db_wo_his_btn = gr.Button("Chat")  # create a button to sunmit questions
            with gr.Row():
                clear = gr.ClearButton(components=[chatbot], value="Clear console")  # create a button to clear the console
                
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot], outputs=[msg, chatbot])  # define what will happen after clicks

    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
gr.close_all()
demo.launch()
```
**We combine the functions defined above to create a `web_demo.py` (Database 1 Local gradio demo) and similarly a  `app.py` (Database 2 OpenXLab)**


### Local the gradio demo with database 1 (InternLM series)

Run `web_demo.py` in the terminal, and Access it http://127.0.0.1:7860 via the **local browser**
```
(langchain) root@intern-studio:~# python data/web_demo.py
正在从本地加载模型...
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:36<00:00,  4.58s/it]
完成本地模型的加载
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

**Test result: The ChatBot knows InternLM but not MMDetection!**
<img src="asserts/InternLM DB1.png" /> </div>


### OpenXLab demo with database 2 (MMDetection and MMEngine)

Some references: https://github.com/Xpg74138/Cattle_lameness_knowledge_assistant, https://github.com/seifer08ms/paper_chat/tree/main

According to the [OpenXLab gradio demo doc](https://openxlab.org.cn/docs/apps/Gradio%E5%BA%94%E7%94%A8.html):
```
├─GitHub repo
│  ├─app.py                       # app.py similar to the web_demo.py
│  ├─requirements.txt             # Python module dependencies
│  ├─packages.txt                 # Debian dependencies（apt-get), can be none
│  └─... 
```

1. Create the `app.py`
According to [OpenXLAB doc](https://openxlab.org.cn/docs/apps/%E5%BA%94%E7%94%A8%E5%88%9B%E5%BB%BA%E6%B5%81%E7%A8%8B.html) and [InternLM-Chat-7B model card](https://openxlab.org.cn/models/detail/OpenLMLab/InternLM-chat-7b), we need to download `Sentence-Transformer` and `InternLM-Chat-7B` in `app.py`
```Python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir /model/sentence-transformer')
```

```Python
from openxlab.model import download
download(model_repo='OpenLMLab/InternLM-chat-7b', output='/model/internlm-chat-7b')
```
```Python
# Change the model path and database path accordingly:

def load_chain():
    """Custom QA chain"""
    embeddings = HuggingFaceEmbeddings(model_name="/model/sentence-transformer")  # embeddings

    persist_directory = 'data_base/vector_db/chroma_2'  # database directory

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)  # get vector database

    llm = InternLM_LLM(model_path="/model/internlm-chat-7b")  # custom LLM

    # Prompt Template
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)  # context and question would be similar text chunck and user prompt in practice

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})  # realize QA using LangChain   
    return qa_chain
```
2. Create the `requirements.txt`
```
pip freeze > requirements.txt
```
We also need to add `pysqlite3-binary `

3. Deploy on OpenXLab
Link: https://openxlab.org.cn/apps/detail/alias-z/langchain_alias-z
<img src="asserts/InternLM DB2 OpenXLab.png" /> </div>
**Test result: Since RAG with only Database 2, the ChatBot did know MMDetection!**