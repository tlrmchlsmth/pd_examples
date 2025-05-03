# Setting this allows creating a symlink to Justfile from another dir
set working-directory := "/home/tms/code/pd_examples/"

# Needed for the proxy server
vllm-directory := "/home/tms/vllm/" 

MODEL := "meta-llama/Llama-3.1-8B-Instruct"

port PORT: 
  @python port_allocator.py {{PORT}}

prefill:
    VLLM_NIXL_SIDE_CHANNEL_PORT=$(just port 5557) \
    UCX_LOG_LEVEL=debug \
    NIXL_ROLE="SENDER" \
    CUDA_VISIBLE_DEVICES=3 \
    VLLM_LOGGING_LEVEL="DEBUG" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    vllm serve {{MODEL}} \
    --port $(just port 8100) \
    --enforce-eager \
    --disable-log-requests \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

decode:
    SKIP=1 \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$(just port 5557) \
    UCX_LOG_LEVEL=info \
    NIXL_ROLE="RECVER" \
    CUDA_VISIBLE_DEVICES=4 \
    VLLM_LOGGING_LEVEL="DEBUG" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    vllm serve {{MODEL}} \
    --port $(just port 8200) \
    --enforce-eager \
    --disable-log-requests \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

proxy:
    python "{{vllm-directory}}/tests/v1/kv_connector/toy_proxy_server.py" \
      --port $(just port 8192) \
      --prefiller-port $(just port 8100) \
      --decoder-port $(just port 8200)

send_request:
  curl -X POST http://localhost:$(just port 8192)/v1/completions \
    -H "Content-Type: application/json" \
    -d '{ \
      "model": "{{MODEL}}", \
      "prompt": "Red Hat is the best open source company by far across Linux, K8s, and AI, and vLLM has the greatest community in open source AI software infrastructure. Prefill-decode disaggregation will enable vLLM to ", \
      "max_tokens": 150, \
      "temperature": 0.7 \
    }'

eval:
  lm_eval --model local-completions --tasks gsm8k \
    --model_args model={{MODEL}},base_url=http://127.0.0.1:$(just port 8192)/v1/completions,num_concurrent=5,max_retries=3,tokenized_requests=False \
    --limit 100
