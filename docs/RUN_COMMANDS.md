# 本地运行操作速查（Mac M4）

你已在M4 MacBook上实际跑通过本地vLLM-Metal、FastAPI、Streamlit和评测脚本。以下是以项目目录为工作目录的启动方式。

### Terminal 1：本地vLLM-Metal（8000）

```bash
~/.venv-vllm-metal/bin/vllm serve mlx-community/Qwen2.5-3B-Instruct-4bit \
  --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.4 --max-model-len 2048
```

第一次启动会下载模型，出现 `Application startup complete.` 后保持该终端运行。

### Terminal 2：FastAPI（8001）

```bash
cd ~/Desktop/enterprise_knowledge_assistant
source .venv/bin/activate
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

测试页面：http://127.0.0.1:8001/docs

### Terminal 3：Streamlit（8501）

```bash
cd ~/Desktop/enterprise_knowledge_assistant
source .venv/bin/activate
python -m streamlit run web_ui.py --server.port 8501
```

聊天界面：http://localhost:8501

### 独立评测（可以先停止其他三个服务以节省内存）

```bash
cd ~/Desktop/enterprise_knowledge_assistant
source .venv/bin/activate
python evaluate.py --validate-only
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false python evaluate.py
```

结果在 `eval_results/summary.json` 和 `eval_results/retrieval_comparison.csv`。

### 常见注意事项

- 需要停止服务：切换到对应终端按 **Control + C**，而不是 Command + C。
- 不要把 `.venv`、本机密钥、Token、付费平台凭证打包上传GitHub。
- `app.py`使用的模型名必须和8000端口vLLM服务的模型名一致。
- Streamlit只是前端；只打开8501但没有运行8000/8001，无法完成问答。
- 目前仅有**固定PDF知识库**；UI不支持用户上传新PDF。
