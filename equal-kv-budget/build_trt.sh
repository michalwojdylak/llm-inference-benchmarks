set -ex
until grep -q DOWNLOAD_DONE ~/rerun/logs/download_bf16.log; do sleep 15; done
cd ~/projects/hosting/tensorrt-llm
source .venv/bin/activate
python ~/TensorRT-LLM/examples/quantization/quantize.py \
  --model_dir models/Qwen3-32B --output_dir Qwen3-32B-FP8-checkpoint \
  --qformat fp8 --dtype bfloat16 --tp_size 1 \
  --kv_cache_dtype fp8 --calib_size 128 --calib_max_seq_length 512 --batch_size 1
python -c "import json;q=json.load(open(\"Qwen3-32B-FP8-checkpoint/config.json\"))[\"quantization\"];print(q);assert \"FP8\" in str(q[\"quant_algo\"]) and \"FP8\" in str(q[\"kv_cache_quant_algo\"])"
rm -rf models/Qwen3-32B/*.safetensors
trtllm-build --checkpoint_dir Qwen3-32B-FP8-checkpoint --output_dir Qwen3-32B-FP8-engine \
  --max_batch_size 64 --max_seq_len 9216 --max_num_tokens 8192 --opt_num_tokens 4096 \
  --gemm_plugin auto --gpt_attention_plugin auto
echo BUILD_TRT_DONE
