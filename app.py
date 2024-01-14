import os
from openxlab.model import download

download(model_repo='alias-z/internlm_chat_7b_qlora_oasst1_e3_alias_z', output='internlm_chat_7b_qlora_oasst1_e3_alias_z')


os.system('streamlit run web_demo.py --server.address=0.0.0.0 --server.port 7860')
