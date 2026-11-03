set -x
uv=~/.local/bin/uv
cd ~/projects/hosting/vllm-workspace && $uv venv --python 3.12 .venv && source .venv/bin/activate && $uv pip install vllm --torch-backend=auto && python -c "import vllm;print(\"vllm\", vllm.__version__)"; deactivate
cd ~/projects/hosting/sglang-workspace && $uv venv --python 3.12 .venv && source .venv/bin/activate && $uv pip install sglang && python -c "import sglang;print(\"sglang\", sglang.__version__)"; deactivate
echo SETUP_ENGINES_DONE
