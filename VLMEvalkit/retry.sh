pkill -f VLLM
pkill -f vllm
sleep 1
nvidia-smi
nohup bash ./run.sh > eval.out 2>&1 &