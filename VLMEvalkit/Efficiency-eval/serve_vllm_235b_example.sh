nohup vllm serve ./Qwen/Qwen3-235B-A22B-Instruct-2507/ \
  --tensor-parallel-size 8 \
  --max-model-len 128000 \
  --max_num_seqs 50 \
  --mm-processor-cache-gb 500 \
  --async-scheduling > ./vllm_server.log 2>&1 &