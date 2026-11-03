set -ex
cd ~/projects/hosting/tensorrt-llm
~/.local/bin/uv venv --python 3.12 .venv
source .venv/bin/activate
~/.local/bin/uv pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu130
~/.local/bin/uv pip install tensorrt_llm==1.2.1 --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
# pyarrow 26 needs numpy 2, which tensorrt_llm 1.2.1 does not allow
~/.local/bin/uv pip install "pyarrow<20"
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
echo SETUP_TRT_DONE
