set -ex
~/.local/bin/uvx --from huggingface_hub hf download Qwen/Qwen3-32B --local-dir ~/projects/hosting/tensorrt-llm/models/Qwen3-32B
echo DOWNLOAD_DONE
