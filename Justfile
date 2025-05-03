prefill:
    UCX_LOG_LEVEL=debug \
    NIXL_ROLE="SENDER" \
    CUDA_VISIBLE_DEVICES=3 \
    VLLM_LOGGING_LEVEL="DEBUG" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8100 \
    --enforce-eager \
    --disable-log-requests \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

decode:
    UCX_LOG_LEVEL=info \
    NIXL_ROLE="RECVER" \
    CUDA_VISIBLE_DEVICES=4 \
    VLLM_LOGGING_LEVEL="DEBUG" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8200 \
    --enforce-eager \
    --disable-log-requests \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

proxy:
    python disagg_proxy_server.py --port 8192

send_request:
  curl -X POST http://localhost:8192/v1/completions \
    -H "Content-Type: application/json" \
    -d '{ \
      "model": "meta-llama/Llama-3.1-8B-Instruct", \
      "prompt": "EXPLAIN KERMIT THE FROG", \
      "max_tokens": 150, \
      "temperature": 0.7 \
    }'

eval:
  lm_eval --model local-completions --tasks gsm8k \
    --model_args model=meta-llama/Llama-3.1-8B-Instruct,base_url=http://127.0.0.1:8192/v1/completions,num_concurrent=5,max_retries=3,tokenized_requests=False \
    --limit 100
