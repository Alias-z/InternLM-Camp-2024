# Assignment 5: OpenCompass
**Tutorial doc**: 
https://github.com/InternLM/tutorial/blob/main/opencompass/opencompass_tutorial.md
<br>
**Tutorial video**: 
https://www.bilibili.com/video/BV1Gg4y1U7uc/?share_source=copy_web&vd_source=7ef00cba9894eae747b373127e0807d4
<br>
**Tasks**:
Deploy InternLM2-Chat-7B via LMDeploy and evaluate it on the `C-Eval` dataset vias OpenCompass


### Install LMDeploy and OpenCompass
1. Start a A100 (1/4) * 2 runtime with `Cuda11.7-conda` docker image on [InternStudio](https://studio.intern-ai.org.cn/)

2. Create a new environment in the terminal (bash):
`conda create --name opencompass --clone=/root/share/conda_envs/internlm-base`

3. After activating the new environment `conda activate opencompass`, we need to install the LMDeploy (version 0.2.0) and OpenCompass:
```
pip install /root/share/wheels/flash_attn-2.4.2+cu118torch2.0cxx11abiTRUE-cp310-cp310-linux_x86_64.whl  # precompiled flash attention
pip install cmake lit packaging 'lmdeploy[all]==v0.2.0'
git clone https://gitee.com/open-compass/opencompass
cd opencompass
pip install -e .
```

### Prepare the dataset
```
cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
unzip OpenCompassData-core-20231110.zip
```

### Test InternLM2-Chat-7B model locally 
```
lmdeploy chat turbomind /share/model_repos/internlm2-chat-7b/  --model-name internlm2-chat-7b
```
The RAM consumption is 36674MiB, much more than that of InternLM-Chat-7B (14854MiB)
```bash
+------------------------------------------------------------------------------+
| VGPU-SMI 1.7.13       Driver Version: 535.54.03     CUDA Version: 12.2       |
+-------------------------------------------+----------------------------------+
| GPU  Name                Bus-Id           |        Memory-Usage     GPU-Util |
|===========================================+==================================|
|   0  NVIDIA A100-SXM...  00000000:4D:00.0 | 36674MiB / 40950MiB    0% /  25% |
+-------------------------------------------+----------------------------------+
```

### Check the supported dataset for the evaluation of InternLM2-Chat-7B
```
cd /root/opencompass
python tools/list_configs.py internlm2 ceval
```
```bash
(opencompass) root@intern-studio:~/opencompass# python tools/list_configs.py internlm2 ceval
+------------------+--------------------------------------------+
| Dataset          | Config Path                                |
|------------------+--------------------------------------------|
| ceval_gen        | configs/datasets/ceval/ceval_gen.py        |
| ceval_gen_2daf24 | configs/datasets/ceval/ceval_gen_2daf24.py |
| ceval_gen_5f30c7 | configs/datasets/ceval/ceval_gen_5f30c7.py |
| ceval_ppl        | configs/datasets/ceval/ceval_ppl.py        |
| ceval_ppl_578f8d | configs/datasets/ceval/ceval_ppl_578f8d.py |
| ceval_ppl_93e5ce | configs/datasets/ceval/ceval_ppl_93e5ce.py |
+------------------+--------------------------------------------+
```

### Evaluate the model on the `ceval_gen`  dataset (objective)
```
python run.py --datasets ceval_gen \
--hf-path /share/model_repos/internlm2-chat-7b/ \
--tokenizer-path /share/model_repos/internlm2-chat-7b/ \
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
--model-kwargs trust_remote_code=True device_map='auto' --max-seq-len 2048 --max-out-len 16 --batch-size 4 --num-gpus 1 \
--debug
```
Evaluation results:
```bash
01/22 21:27:08 - OpenCompass - DEBUG - An `DefaultSummarizer` instance is built from registry, and its implementation can be found in opencompass.summarizers.default
dataset                                         version    metric         mode      opencompass.models.huggingface.HuggingFace_model_repos_internlm2-chat-7b
----------------------------------------------  ---------  -------------  ------  --------------------------------------------------------------------------
ceval-computer_network                          db9ce2     accuracy       gen                                                                          47.37
ceval-operating_system                          1c2571     accuracy       gen                                                                          57.89
ceval-computer_architecture                     a74dad     accuracy       gen                                                                          38.1
ceval-college_programming                       4ca32a     accuracy       gen                                                                          18.92
ceval-college_physics                           963fa8     accuracy       gen                                                                           5.26
ceval-college_chemistry                         e78857     accuracy       gen                                                                           0
ceval-advanced_mathematics                      ce03e2     accuracy       gen                                                                           0
ceval-probability_and_statistics                65e812     accuracy       gen                                                                          11.11
ceval-discrete_mathematics                      e894ae     accuracy       gen                                                                          18.75
ceval-electrical_engineer                       ae42b9     accuracy       gen                                                                          18.92
ceval-metrology_engineer                        ee34ea     accuracy       gen                                                                          50
ceval-high_school_mathematics                   1dc5bf     accuracy       gen                                                                           0
ceval-high_school_physics                       adf25f     accuracy       gen                                                                          31.58
ceval-high_school_chemistry                     2ed27f     accuracy       gen                                                                          26.32
ceval-high_school_biology                       8e2b9a     accuracy       gen                                                                          26.32
ceval-middle_school_mathematics                 bee8d5     accuracy       gen                                                                          21.05
ceval-middle_school_biology                     86817c     accuracy       gen                                                                          66.67
ceval-middle_school_physics                     8accf6     accuracy       gen                                                                          52.63
ceval-middle_school_chemistry                   167a15     accuracy       gen                                                                          80
ceval-veterinary_medicine                       b4e08d     accuracy       gen                                                                          39.13
ceval-college_economics                         f3f4e6     accuracy       gen                                                                          29.09
ceval-business_administration                   c1614e     accuracy       gen                                                                          30.3
ceval-marxism                                   cf874c     accuracy       gen                                                                          84.21
ceval-mao_zedong_thought                        51c7a4     accuracy       gen                                                                          70.83
ceval-education_science                         591fee     accuracy       gen                                                                          62.07
ceval-teacher_qualification                     4e4ced     accuracy       gen                                                                          77.27
ceval-high_school_politics                      5c0de2     accuracy       gen                                                                          21.05
ceval-high_school_geography                     865461     accuracy       gen                                                                          47.37
ceval-middle_school_politics                    5be3e7     accuracy       gen                                                                          38.1
ceval-middle_school_geography                   8a63be     accuracy       gen                                                                          58.33
ceval-modern_chinese_history                    fc01af     accuracy       gen                                                                          65.22
ceval-ideological_and_moral_cultivation         a2aa4a     accuracy       gen                                                                          89.47
ceval-logic                                     f5b022     accuracy       gen                                                                          13.64
ceval-law                                       a110a1     accuracy       gen                                                                          37.5
ceval-chinese_language_and_literature           0f8b68     accuracy       gen                                                                          47.83
ceval-art_studies                               2a1300     accuracy       gen                                                                          66.67
ceval-professional_tour_guide                   4e673e     accuracy       gen                                                                          82.76
ceval-legal_professional                        ce8787     accuracy       gen                                                                          30.43
ceval-high_school_chinese                       315705     accuracy       gen                                                                          21.05
ceval-high_school_history                       7eb30a     accuracy       gen                                                                          75
ceval-middle_school_history                     48ab4a     accuracy       gen                                                                          68.18
ceval-civil_servant                             87d061     accuracy       gen                                                                          38.3
ceval-sports_science                            70f27b     accuracy       gen                                                                          63.16
ceval-plant_protection                          8941f9     accuracy       gen                                                                          68.18
ceval-basic_medicine                            c409d6     accuracy       gen                                                                          57.89
ceval-clinical_medicine                         49e82d     accuracy       gen                                                                          45.45
ceval-urban_and_rural_planner                   95b885     accuracy       gen                                                                          58.7
ceval-accountant                                002837     accuracy       gen                                                                          34.69
ceval-fire_engineer                             bc23f5     accuracy       gen                                                                          12.9
ceval-environmental_impact_assessment_engineer  c64e2d     accuracy       gen                                                                          38.71
ceval-tax_accountant                            3a5e3c     accuracy       gen                                                                          42.86
ceval-physician                                 6e277d     accuracy       gen                                                                          51.02
ceval-stem                                      -          naive_average  gen                                                                          30.5
ceval-social-science                            -          naive_average  gen                                                                          51.86
ceval-humanities                                -          naive_average  gen                                                                          54.34
ceval-other                                     -          naive_average  gen                                                                          46.53
ceval-hard                                      -          naive_average  gen                                                                          11.63
ceval                                           -          naive_average  gen                                                                          43.04
01/22 21:27:08 - OpenCompass - INFO - write summary to /root/opencompass/outputs/default/20240122_201352/summary/summary_20240122_201352.txt
01/22 21:27:08 - OpenCompass - INFO - write csv to /root/opencompass/outputs/default/20240122_201352/summary/summary_20240122_201352.csv
```

### Apply W4A16 on InternLM2-Chat-7B and evaluate it

**References**: https://zhuanlan.zhihu.com/p/678558095, https://github.com/Shengshenlan/tutorial/blob/master/homework/6/%E7%AC%AC6%E8%8A%82%E8%AF%BE%E4%BD%9C%E4%B8%9A.md

1. Apply W4A16 on InternLM2-Chat-7B

```
lmdeploy lite auto_awq /root/share/model_repos/internlm2-chat-7b  --work-dir /root/internlm2-chat-7b-4bit
lmdeploy convert  internlm2-chat-7b /root/internlm2-chat-7b-4bit/ --model-format awq --group-size 128  --dst-path  /root/workspace_awq
lmdeploy chat turbomind /root/workspace_awq
```
The RAM consumption is reduced to 26820MiB from 36674MiB

```bash
+------------------------------------------------------------------------------+
| VGPU-SMI 1.7.13       Driver Version: 535.54.03     CUDA Version: 12.2       |
+-------------------------------------------+----------------------------------+
| GPU  Name                Bus-Id           |        Memory-Usage     GPU-Util |
|===========================================+==================================|
|   0  NVIDIA A100-SXM...  00000000:4D:00.0 | 26820MiB / 40950MiB    0% /  25% |
+-------------------------------------------+----------------------------------+
```
3. Make a copy of `/root/opencompass/configs/eval_internlm_chat_turbomind.py` and rename it as `eval_internlm2_chat_turbomind.py`, and make modifications
```Python
from mmengine.config import read_base
from opencompass.models.turbomind import TurboMindModel

with read_base():
    # choose a list of datasets
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    # and output the results in a choosen format
    from .summarizers.medium import summarizer


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

internlm_meta_template = dict(round=[
    dict(role='HUMAN', begin='<|User|>:', end='\n'),
    dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
],
                              eos_token_id=103028)

# config for internlm-chat-7b
internlm2_chat_7b = dict(
    type=TurboMindModel,
    abbr='internlm2-chat-7b-turbomind',
    path='/root/workspace_awq',
    engine_config=dict(session_len=2048,
                       max_batch_size=32,
                       rope_scaling_factor=1.0),
    gen_config=dict(top_k=1,
                    top_p=0.8,
                    temperature=1.0,
                    max_new_tokens=100),
    max_out_len=100,
    max_seq_len=2048,
    batch_size=32,
    concurrency=32,
    meta_template=internlm_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

models = [internlm2_chat_7b]
```
Evaluate the quantized model
```
cd /root/opencompass
python run.py /root/opencompass/configs/eval_internlm2_chat_turbomind.py --debug
```
Evaluation results:
It seems that the results got better after quantization, but note that `max_out_len` and `max_seq_len` were set differently as well. For a fair comparison, parameters should be set as the same. However, no further experiment is planned for this assignment due to time limitations. 
```bash
01/23 01:55:47 - OpenCompass - INFO - Partitioned into 52 tasks.
01/23 01:55:47 - OpenCompass - DEBUG - Task 0: [internlm2-chat-7b-turbomind/ceval-computer_network]
01/23 01:55:47 - OpenCompass - DEBUG - Task 1: [internlm2-chat-7b-turbomind/ceval-operating_system]
01/23 01:55:47 - OpenCompass - DEBUG - Task 2: [internlm2-chat-7b-turbomind/ceval-computer_architecture]
01/23 01:55:47 - OpenCompass - DEBUG - Task 3: [internlm2-chat-7b-turbomind/ceval-college_programming]
01/23 01:55:47 - OpenCompass - DEBUG - Task 4: [internlm2-chat-7b-turbomind/ceval-college_physics]
01/23 01:55:47 - OpenCompass - DEBUG - Task 5: [internlm2-chat-7b-turbomind/ceval-college_chemistry]
01/23 01:55:47 - OpenCompass - DEBUG - Task 6: [internlm2-chat-7b-turbomind/ceval-advanced_mathematics]
01/23 01:55:47 - OpenCompass - DEBUG - Task 7: [internlm2-chat-7b-turbomind/ceval-probability_and_statistics]
01/23 01:55:47 - OpenCompass - DEBUG - Task 8: [internlm2-chat-7b-turbomind/ceval-discrete_mathematics]
01/23 01:55:47 - OpenCompass - DEBUG - Task 9: [internlm2-chat-7b-turbomind/ceval-electrical_engineer]
01/23 01:55:47 - OpenCompass - DEBUG - Task 10: [internlm2-chat-7b-turbomind/ceval-metrology_engineer]
01/23 01:55:47 - OpenCompass - DEBUG - Task 11: [internlm2-chat-7b-turbomind/ceval-high_school_mathematics]
01/23 01:55:47 - OpenCompass - DEBUG - Task 12: [internlm2-chat-7b-turbomind/ceval-high_school_physics]
01/23 01:55:47 - OpenCompass - DEBUG - Task 13: [internlm2-chat-7b-turbomind/ceval-high_school_chemistry]
01/23 01:55:47 - OpenCompass - DEBUG - Task 14: [internlm2-chat-7b-turbomind/ceval-high_school_biology]
01/23 01:55:47 - OpenCompass - DEBUG - Task 15: [internlm2-chat-7b-turbomind/ceval-middle_school_mathematics]
01/23 01:55:47 - OpenCompass - DEBUG - Task 16: [internlm2-chat-7b-turbomind/ceval-middle_school_biology]
01/23 01:55:47 - OpenCompass - DEBUG - Task 17: [internlm2-chat-7b-turbomind/ceval-middle_school_physics]
01/23 01:55:47 - OpenCompass - DEBUG - Task 18: [internlm2-chat-7b-turbomind/ceval-middle_school_chemistry]
01/23 01:55:47 - OpenCompass - DEBUG - Task 19: [internlm2-chat-7b-turbomind/ceval-veterinary_medicine]
01/23 01:55:47 - OpenCompass - DEBUG - Task 20: [internlm2-chat-7b-turbomind/ceval-college_economics]
01/23 01:55:47 - OpenCompass - DEBUG - Task 21: [internlm2-chat-7b-turbomind/ceval-business_administration]
01/23 01:55:47 - OpenCompass - DEBUG - Task 22: [internlm2-chat-7b-turbomind/ceval-marxism]
01/23 01:55:47 - OpenCompass - DEBUG - Task 23: [internlm2-chat-7b-turbomind/ceval-mao_zedong_thought]
01/23 01:55:47 - OpenCompass - DEBUG - Task 24: [internlm2-chat-7b-turbomind/ceval-education_science]
01/23 01:55:47 - OpenCompass - DEBUG - Task 25: [internlm2-chat-7b-turbomind/ceval-teacher_qualification]
01/23 01:55:47 - OpenCompass - DEBUG - Task 26: [internlm2-chat-7b-turbomind/ceval-high_school_politics]
01/23 01:55:47 - OpenCompass - DEBUG - Task 27: [internlm2-chat-7b-turbomind/ceval-high_school_geography]
01/23 01:55:47 - OpenCompass - DEBUG - Task 28: [internlm2-chat-7b-turbomind/ceval-middle_school_politics]
01/23 01:55:47 - OpenCompass - DEBUG - Task 29: [internlm2-chat-7b-turbomind/ceval-middle_school_geography]
01/23 01:55:47 - OpenCompass - DEBUG - Task 30: [internlm2-chat-7b-turbomind/ceval-modern_chinese_history]
01/23 01:55:47 - OpenCompass - DEBUG - Task 31: [internlm2-chat-7b-turbomind/ceval-ideological_and_moral_cultivation]
01/23 01:55:47 - OpenCompass - DEBUG - Task 32: [internlm2-chat-7b-turbomind/ceval-logic]
01/23 01:55:47 - OpenCompass - DEBUG - Task 33: [internlm2-chat-7b-turbomind/ceval-law]
01/23 01:55:47 - OpenCompass - DEBUG - Task 34: [internlm2-chat-7b-turbomind/ceval-chinese_language_and_literature]
01/23 01:55:47 - OpenCompass - DEBUG - Task 35: [internlm2-chat-7b-turbomind/ceval-art_studies]
01/23 01:55:47 - OpenCompass - DEBUG - Task 36: [internlm2-chat-7b-turbomind/ceval-professional_tour_guide]
01/23 01:55:47 - OpenCompass - DEBUG - Task 37: [internlm2-chat-7b-turbomind/ceval-legal_professional]
01/23 01:55:47 - OpenCompass - DEBUG - Task 38: [internlm2-chat-7b-turbomind/ceval-high_school_chinese]
01/23 01:55:47 - OpenCompass - DEBUG - Task 39: [internlm2-chat-7b-turbomind/ceval-high_school_history]
01/23 01:55:47 - OpenCompass - DEBUG - Task 40: [internlm2-chat-7b-turbomind/ceval-middle_school_history]
01/23 01:55:47 - OpenCompass - DEBUG - Task 41: [internlm2-chat-7b-turbomind/ceval-civil_servant]
01/23 01:55:47 - OpenCompass - DEBUG - Task 42: [internlm2-chat-7b-turbomind/ceval-sports_science]
01/23 01:55:47 - OpenCompass - DEBUG - Task 43: [internlm2-chat-7b-turbomind/ceval-plant_protection]
01/23 01:55:47 - OpenCompass - DEBUG - Task 44: [internlm2-chat-7b-turbomind/ceval-basic_medicine]
01/23 01:55:47 - OpenCompass - DEBUG - Task 45: [internlm2-chat-7b-turbomind/ceval-clinical_medicine]
01/23 01:55:47 - OpenCompass - DEBUG - Task 46: [internlm2-chat-7b-turbomind/ceval-urban_and_rural_planner]
01/23 01:55:47 - OpenCompass - DEBUG - Task 47: [internlm2-chat-7b-turbomind/ceval-accountant]
01/23 01:55:47 - OpenCompass - DEBUG - Task 48: [internlm2-chat-7b-turbomind/ceval-fire_engineer]
01/23 01:55:47 - OpenCompass - DEBUG - Task 49: [internlm2-chat-7b-turbomind/ceval-environmental_impact_assessment_engineer]
01/23 01:55:47 - OpenCompass - DEBUG - Task 50: [internlm2-chat-7b-turbomind/ceval-tax_accountant]
01/23 01:55:47 - OpenCompass - DEBUG - Task 51: [internlm2-chat-7b-turbomind/ceval-physician]
01/23 01:55:47 - OpenCompass - DEBUG - Get class `LocalRunner` from "runner" registry in "opencompass"
01/23 01:55:47 - OpenCompass - DEBUG - An `LocalRunner` instance is built from registry, and its implementation can be found in opencompass.runners.local
01/23 01:55:47 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 01:55:47 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 01:56:46 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-computer_network]: {'accuracy': 52.63157894736842}
01/23 01:56:46 - OpenCompass - INFO - time elapsed: 9.49s
01/23 01:56:47 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 01:56:47 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 01:56:56 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-operating_system]: {'accuracy': 63.1578947368421}
01/23 01:56:56 - OpenCompass - INFO - time elapsed: 3.65s
01/23 01:56:56 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 01:56:56 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 01:57:07 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-computer_architecture]: {'accuracy': 52.38095238095239}
01/23 01:57:07 - OpenCompass - INFO - time elapsed: 5.12s
01/23 01:57:08 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 01:57:08 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 01:57:19 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-college_programming]: {'accuracy': 54.054054054054056}
01/23 01:57:19 - OpenCompass - INFO - time elapsed: 5.36s
01/23 01:57:20 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 01:57:20 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 01:57:30 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-college_physics]: {'accuracy': 42.10526315789473}
01/23 01:57:30 - OpenCompass - INFO - time elapsed: 4.18s
01/23 01:57:30 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 01:57:30 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 01:58:23 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-college_chemistry]: {'accuracy': 33.33333333333333}
01/23 01:58:23 - OpenCompass - INFO - time elapsed: 46.99s
01/23 01:58:24 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 01:58:24 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 01:58:35 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-advanced_mathematics]: {'accuracy': 42.10526315789473}
01/23 01:58:35 - OpenCompass - INFO - time elapsed: 5.38s
01/23 01:58:36 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 01:58:36 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 01:58:46 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-probability_and_statistics]: {'accuracy': 44.44444444444444}
01/23 01:58:46 - OpenCompass - INFO - time elapsed: 4.58s
01/23 01:58:47 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 01:58:47 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 01:58:57 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-discrete_mathematics]: {'accuracy': 25.0}
01/23 01:58:57 - OpenCompass - INFO - time elapsed: 4.74s
01/23 01:58:58 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 01:58:58 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 01:59:08 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-electrical_engineer]: {'accuracy': 43.24324324324324}
01/23 01:59:08 - OpenCompass - INFO - time elapsed: 5.04s
01/23 01:59:09 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 01:59:09 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 01:59:37 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-metrology_engineer]: {'accuracy': 66.66666666666666}
01/23 01:59:39 - OpenCompass - INFO - time elapsed: 8.98s
01/23 01:59:40 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 01:59:40 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 01:59:51 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-high_school_mathematics]: {'accuracy': 44.44444444444444}
01/23 01:59:51 - OpenCompass - INFO - time elapsed: 5.38s
01/23 01:59:52 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 01:59:52 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:00:32 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-high_school_physics]: {'accuracy': 42.10526315789473}
01/23 02:00:32 - OpenCompass - INFO - time elapsed: 35.08s
01/23 02:00:33 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:00:33 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:00:50 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-high_school_chemistry]: {'accuracy': 36.84210526315789}
01/23 02:01:23 - OpenCompass - INFO - time elapsed: 41.28s
01/23 02:01:24 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:01:24 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:01:40 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-high_school_biology]: {'accuracy': 36.84210526315789}
01/23 02:01:40 - OpenCompass - INFO - time elapsed: 9.28s
01/23 02:01:41 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:01:41 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:02:00 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-middle_school_mathematics]: {'accuracy': 42.10526315789473}
01/23 02:02:21 - OpenCompass - INFO - time elapsed: 32.51s
01/23 02:02:21 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:02:21 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:03:26 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-middle_school_biology]: {'accuracy': 76.19047619047619}
01/23 02:03:26 - OpenCompass - INFO - time elapsed: 43.53s
01/23 02:03:27 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:03:27 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:03:39 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-middle_school_physics]: {'accuracy': 57.89473684210527}
01/23 02:04:21 - OpenCompass - INFO - time elapsed: 47.61s
01/23 02:04:22 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:04:22 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:05:18 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-middle_school_chemistry]: {'accuracy': 90.0}
01/23 02:05:18 - OpenCompass - INFO - time elapsed: 5.21s
01/23 02:05:19 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:05:19 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:05:59 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-veterinary_medicine]: {'accuracy': 52.17391304347826}
01/23 02:05:59 - OpenCompass - INFO - time elapsed: 34.27s
01/23 02:05:59 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:05:59 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:06:11 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-college_economics]: {'accuracy': 47.27272727272727}
01/23 02:06:11 - OpenCompass - INFO - time elapsed: 5.47s
01/23 02:06:11 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:06:11 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:07:12 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-business_administration]: {'accuracy': 54.54545454545454}
01/23 02:07:12 - OpenCompass - INFO - time elapsed: 6.24s
01/23 02:07:12 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:07:12 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:07:24 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-marxism]: {'accuracy': 78.94736842105263}
01/23 02:07:24 - OpenCompass - INFO - time elapsed: 5.34s
01/23 02:07:24 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:07:24 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:08:25 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-mao_zedong_thought]: {'accuracy': 75.0}
01/23 02:08:25 - OpenCompass - INFO - time elapsed: 42.14s
01/23 02:08:26 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:08:26 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:09:19 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-education_science]: {'accuracy': 65.51724137931035}
01/23 02:09:19 - OpenCompass - INFO - time elapsed: 47.59s
01/23 02:09:20 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:09:20 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:09:31 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-teacher_qualification]: {'accuracy': 79.54545454545455}
01/23 02:10:19 - OpenCompass - INFO - time elapsed: 53.20s
01/23 02:10:20 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:10:20 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:11:19 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-high_school_politics]: {'accuracy': 89.47368421052632}
01/23 02:12:03 - OpenCompass - INFO - time elapsed: 97.01s
01/23 02:12:03 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:12:03 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:12:28 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-high_school_geography]: {'accuracy': 57.89473684210527}
01/23 02:12:28 - OpenCompass - INFO - time elapsed: 18.48s
01/23 02:12:28 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:12:28 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:12:40 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-middle_school_politics]: {'accuracy': 76.19047619047619}
01/23 02:12:40 - OpenCompass - INFO - time elapsed: 5.49s
01/23 02:12:41 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:12:41 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:12:51 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-middle_school_geography]: {'accuracy': 83.33333333333334}
01/23 02:12:51 - OpenCompass - INFO - time elapsed: 5.00s
01/23 02:12:52 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:12:52 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:13:03 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-modern_chinese_history]: {'accuracy': 69.56521739130434}
01/23 02:13:48 - OpenCompass - INFO - time elapsed: 50.56s
01/23 02:13:49 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:13:49 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:14:00 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-ideological_and_moral_cultivation]: {'accuracy': 73.68421052631578}
01/23 02:14:00 - OpenCompass - INFO - time elapsed: 5.13s
01/23 02:14:01 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:14:01 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:14:10 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-logic]: {'accuracy': 54.54545454545454}
01/23 02:14:10 - OpenCompass - INFO - time elapsed: 4.57s
01/23 02:14:11 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:14:11 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:14:20 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-law]: {'accuracy': 45.83333333333333}
01/23 02:14:20 - OpenCompass - INFO - time elapsed: 4.24s
01/23 02:14:21 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:14:21 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:14:32 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-chinese_language_and_literature]: {'accuracy': 56.52173913043478}
01/23 02:14:32 - OpenCompass - INFO - time elapsed: 5.02s
01/23 02:14:32 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:14:32 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:14:42 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-art_studies]: {'accuracy': 66.66666666666666}
01/23 02:14:42 - OpenCompass - INFO - time elapsed: 4.48s
01/23 02:14:43 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:14:43 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:14:53 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-professional_tour_guide]: {'accuracy': 82.75862068965517}
01/23 02:14:53 - OpenCompass - INFO - time elapsed: 4.50s
01/23 02:14:53 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:14:53 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:15:05 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-legal_professional]: {'accuracy': 43.47826086956522}
01/23 02:15:52 - OpenCompass - INFO - time elapsed: 53.08s
01/23 02:15:53 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:15:53 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:16:59 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-high_school_chinese]: {'accuracy': 47.368421052631575}
01/23 02:16:59 - OpenCompass - INFO - time elapsed: 46.54s
01/23 02:16:59 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:16:59 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:17:36 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-high_school_history]: {'accuracy': 75.0}
01/23 02:17:36 - OpenCompass - INFO - time elapsed: 29.74s
01/23 02:17:37 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:17:37 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:18:28 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-middle_school_history]: {'accuracy': 77.27272727272727}
01/23 02:18:28 - OpenCompass - INFO - time elapsed: 46.16s
01/23 02:18:29 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:18:29 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:18:49 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-civil_servant]: {'accuracy': 59.57446808510638}
01/23 02:18:49 - OpenCompass - INFO - time elapsed: 13.80s
01/23 02:18:50 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:18:50 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:19:00 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-sports_science]: {'accuracy': 63.1578947368421}
01/23 02:19:00 - OpenCompass - INFO - time elapsed: 4.55s
01/23 02:19:00 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:19:00 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:19:11 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-plant_protection]: {'accuracy': 59.09090909090909}
01/23 02:19:11 - OpenCompass - INFO - time elapsed: 4.81s
01/23 02:19:12 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:19:12 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:19:28 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-basic_medicine]: {'accuracy': 78.94736842105263}
01/23 02:19:28 - OpenCompass - INFO - time elapsed: 10.20s
01/23 02:19:29 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:19:29 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:20:21 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-clinical_medicine]: {'accuracy': 54.54545454545454}
01/23 02:20:21 - OpenCompass - INFO - time elapsed: 46.56s
01/23 02:20:21 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:20:21 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:21:43 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-urban_and_rural_planner]: {'accuracy': 58.69565217391305}
01/23 02:21:43 - OpenCompass - INFO - time elapsed: 59.25s
01/23 02:21:44 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:21:44 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:21:54 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-accountant]: {'accuracy': 51.02040816326531}
01/23 02:21:54 - OpenCompass - INFO - time elapsed: 4.51s
01/23 02:21:55 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:21:55 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:22:04 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-fire_engineer]: {'accuracy': 38.70967741935484}
01/23 02:22:04 - OpenCompass - INFO - time elapsed: 3.77s
01/23 02:22:04 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:22:04 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:22:14 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-environmental_impact_assessment_engineer]: {'accuracy': 51.61290322580645}
01/23 02:22:14 - OpenCompass - INFO - time elapsed: 4.31s
01/23 02:22:14 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:22:14 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:22:24 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-tax_accountant]: {'accuracy': 57.14285714285714}
01/23 02:22:24 - OpenCompass - INFO - time elapsed: 4.37s
01/23 02:22:24 - OpenCompass - DEBUG - Get class `OpenICLEvalTask` from "task" registry in "opencompass"
01/23 02:22:24 - OpenCompass - DEBUG - An `OpenICLEvalTask` instance is built from registry, and its implementation can be found in opencompass.tasks.openicl_eval
01/23 02:22:34 - OpenCompass - INFO - Task [internlm2-chat-7b-turbomind/ceval-physician]: {'accuracy': 53.06122448979592}
01/23 02:22:34 - OpenCompass - INFO - time elapsed: 4.24s
01/23 02:22:35 - OpenCompass - DEBUG - An `DefaultSummarizer` instance is built from registry, and its implementation can be found in opencompass.summarizers.default

01/23 02:22:35 - OpenCompass - INFO - write summary to /root/opencompass/outputs/default/20240123_015110/summary/summary_20240123_015110.txt
01/23 02:22:35 - OpenCompass - INFO - write csv to /root/opencompass/outputs/default/20240123_015110/summary/summary_20240123_015110.csv
```

### Error handling

The following error occurred, leading to no results in the predictions
```bash
Traceback (most recent call last):
  File "/root/opencompass/opencompass/tasks/openicl_infer.py", line 148, in <module>
    inferencer.run()
  File "/root/opencompass/opencompass/tasks/openicl_infer.py", line 78, in run
    self._inference()
  File "/root/opencompass/opencompass/tasks/openicl_infer.py", line 126, in _inference
    inferencer.inference(retriever,
  File "/root/opencompass/opencompass/openicl/icl_inferencer/icl_gen_inferencer.py", line 89, in inference
    prompt_list = self.get_generation_prompt_list_from_retriever_indices(
  File "/root/opencompass/opencompass/openicl/icl_inferencer/icl_gen_inferencer.py", line 180, in get_generation_prompt_list_from_retriever_indices
    while len(ice_idx) > 0 and prompt_token_num > max_seq_len:
TypeError: '>' not supported between instances of 'NoneType' and 'int'
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 340026) of binary: /root/.conda/envs/opencompass/bin/python
Traceback (most recent call last):
  File "/root/.conda/envs/opencompass/bin/torchrun", line 33, in <module>
    sys.exit(load_entry_point('torch==2.0.1', 'console_scripts', 'torchrun')())
  File "/root/.conda/envs/opencompass/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 346, in wrapper
    return f(*args, **kwargs)
  File "/root/.conda/envs/opencompass/lib/python3.10/site-packages/torch/distributed/run.py", line 794, in main
    run(args)
  File "/root/.conda/envs/opencompass/lib/python3.10/site-packages/torch/distributed/run.py", line 785, in run
    elastic_launch(
  File "/root/.conda/envs/opencompass/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 134, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/root/.conda/envs/opencompass/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 250, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
/root/opencompass/opencompass/tasks/openicl_infer.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2024-01-23_00:54:00
  host      : intern-studio
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 340026)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
```

Solutions: STOP USING GITEE! Upload `OpenCompass.zip` that was downloaded from GitHub
The two versions are essentially different
<img src="asserts/git vs gitee.png" /> </div>

