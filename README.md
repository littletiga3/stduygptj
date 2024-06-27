# stduygptj
Self study on how to fine tune LLM

启动命令

`accelerate launch --dynamo_backend=inductor --num_processes=1 --num_machines=1 --machine_rank=0 --mixed_precision=bf16 train.py --config config.yaml`
