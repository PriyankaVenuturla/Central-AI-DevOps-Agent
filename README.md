# Central-Agent: AI-Powered CI/CD Failure Triage Agent

A centralized, enterprise-grade AI triage service built on **AWS Bedrock (`moonshotai.kimi-k2.5`)**, **LangGraph**, and **ChromaDB (`all-MiniLM-L6-v2`)**.

Based on the architectural specifications from `AI.pptx`.

---

## 🎯 Project Intent & Core Invariants

1. **Triggered on Failure Only ($0 Compute on Passing Builds):**  
   Listens to GitHub webhook `workflow_run.completed` events and only executes when `conclusion == "failure"`.
2. **Noise Pruning & 50-Line Boundary Chunking:**  
   Cleans up 10,000+ lines of messy terminal output down to the exact 50-line error perimeter around the failure marker.
3. **Institutional Memory (ChromaDB + all-MiniLM-L6-v2):**  
   Semantic vector lookup against team runbooks and past solved issues to suggest proven resolutions.
4. **AWS Bedrock LLM (`moonshotai.kimi-k2.5`):**  
   Converse API integration with prompt injection defense, secret redaction, and structured Pydantic analysis.
5. **Strict Zero-Code-Mutation Safety Invariant:**  
   The agent **only triages and alerts** in plain English—it never alters production branches or merges pull requests.

---

## 📁 Modular Code Architecture

```
Central-Agent/
├── config.py                 # Configuration & secrets management (Bedrock token, GitHub, Chroma)
├── models.py                 # Pydantic schemas (AnalysisResult, JobLogData, WebhookRunEvent)
│
├── core/
│   ├── log_filter.py         # Noise pruning regex & 50-line boundary chunker
│   ├── guardrails.py         # Pre-LLM secret redactor & prompt injection XML sandbox
│   └── bedrock_llm.py        # AWS Bedrock Converse client for moonshotai.kimi-k2.5
│
├── memory/
│   └── chroma_memory.py      # ChromaDB vector store with all-MiniLM-L6-v2 embeddings
│
├── tools/
│   ├── github_client.py      # GitHub API client (fetching runs, logs, source, posting PR comments)
│   └── mcp_interface.py      # Model Context Protocol (MCP) standardized tool contracts
│
├── graph/
│   ├── state.py              # LangGraph AgentState dictionary
│   ├── nodes.py              # Discrete LangGraph node implementations
│   └── workflow.py           # StateGraph assembly, compilation & sequential fallback
│
├── webhook_server.py         # FastAPI webhook listener with HMAC SHA-256 verification
├── run_triage.py             # CLI entrypoint for direct run or local log triage
├── test_agent.py             # End-to-end integration and component test suite
├── requirements.txt          # Python dependencies
└── .env.example              # Environment variables template
```

---

## 🚀 Quickstart & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` or set in your environment:
```bash
export AWS_BEARER_TOKEN_BEDROCK="<your-bedrock-token>"
export AWS_DEFAULT_REGION="ap-south-1"
export BEDROCK_MODEL_ID="moonshotai.kimi-k2.5"
export GITHUB_TOKEN="<ghp_token>"
```

### 3. Run Self-Tests
```bash
python test_agent.py
```

### 4. Run CLI Triage on a Log File
```bash
python run_triage.py --repo "my-org/my-service" --run-id 12345 --log-file path/to/failed_build.log
```

### 5. Start the Webhook Listener (FastAPI)
```bash
python webhook_server.py
```
Endpoint available at `http://localhost:8000/webhook`. Configure this URL in your GitHub Organization or Repository Webhook settings.
