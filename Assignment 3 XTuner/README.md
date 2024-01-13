# Assignment 3: XTuner Single-GPU Finetuning
**Tutorial doc**: https://github.com/InternLM/tutorial/blob/main/xtuner/README.md
<br>
**Tutorial video**: https://www.bilibili.com/video/BV1yK4y1B75J/?share_source=copy_web&vd_source=7ef00cba9894eae747b373127e0807d4
<br>
**Tasks**:
1. Create a training set so that the model knows he works specifically for me (Alias-z)
2. Upload the adaptor model and deploy the model on OpenXLab, according to the [doc](https://aicarrier.feishu.cn/docx/MQH6dygcKolG37x0ekcc4oZhnCe)


### Install XTuner
1. Start a A100 (1/4) runtime with `Cuda11.7-conda` docker image on [InternStudio](https://studio.intern-ai.org.cn/)

2. Create a new environment in the terminal (bash):
`conda create --name xtuner python=3.10 -y`

3. After activating the new environment `conda activate xtuner`, we need to install the XTuner0.1.9:
```
mkdir -p /root/xtuner019
cd /root/xtuner019
git clone https://github.com/InternLM/xtuner -b v0.1.9
cd xtuner
pip install -e '.[all]'
```
4. (Optional) Update `apt` and install `tmux` to prevent training interruption
```
apt update -y
apt install tmux -y
tmux new -s xtuner

tmux attach -t xtuner  # switch back to tmux mode or to bash (Control + B D)
```

### Prepare dataset and configs

1. Prepare the directory for `oasst1` dataset, which will be used for finetuning
```
mkdir -p /root/ft-oasst1
cd /root/ft-oasst1
```

2. Fetch the `InternLM-Chat-7B-QLora-Oasst1` config, model weights and dataset
```
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
cp -r /root/share/temp/model_repos/internlm-chat-7b /root/ft-oasst1/
cp -r /root/share/temp/datasets/openassistant-guanaco .
```

3. Modify `internlm_chat_7b_qlora_oasst1_e3_copy.py`
```
pretrained_model_name_or_path = './internlm-chat-7b'

data_path = './openassistant-guanaco'

epoch = 1

evaluation_inputs = 
```

### Finetuning

1. Start finetuning with DeepSpeed
```
xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py --deepspeed deepspeed_zero2
```

2. Convert adaptor weight from `.pth` to `.hf`
```
cd /root/ft-oasst1/
mkdir hf
export MKL_SERVICE_FORCE_INTEL=1

xtuner convert pth_to_hf ./internlm_chat_7b_qlora_oasst1_e3_copy.py ./work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy/epoch_1.pth ./hf
```

3. Merged the `.hf` adaptor weights to the base model
```
xtuner convert merge ./internlm-chat-7b ./hf ./merged --max-shard-size 2GB
```

4. Test the merged model
```
xtuner chat ./merged --prompt-template internlm_chat

xtuner chat ./merged --bits 4 --prompt-template internlm_chat # 4-bit mmode
```