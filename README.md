# Zenith AI - Intelligent Front Desk Agent

<div align="center">

**An intelligent multi-agent system for healthcare front desk operations, powered by LangGraph, RAG, and advanced ML models.**

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [API Documentation](#-api-documentation) • [Development](#-development)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Development](#-development)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

Zenith AI is a production-ready intelligent agent system designed for healthcare front desk operations. It combines the power of Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and multi-agent orchestration to handle inquiries, bookings, and database queries with human-like understanding and efficiency.

### Key Capabilities

- **🤖 Multi-Agent Orchestration**: Intelligent routing and task delegation using LangGraph
- **🔍 Semantic Search**: Advanced RAG with vector embeddings and XGBoost reranking
- **💬 Natural Conversations**: Context-aware chat interface with streaming responses
- **📅 Booking Management**: Automated appointment scheduling and management
- **🗄️ Database Integration**: Direct SQL query execution for real-time data access
- **📊 ML Model Tracking**: MLflow integration for model versioning and monitoring

## ✨ Features

### Core Features

- **Intelligent Routing**: Manager agent intelligently routes queries to specialized agents
- **RAG-Powered Knowledge Base**: Semantic search over document collections with reranking
- **Streaming Responses**: Real-time streaming for responsive user experience
- **Multi-Thread Conversations**: Session management with thread-based context
- **Model Versioning**: MLflow integration for tracking and deploying ML models
- **Dockerized Deployment**: Complete containerization with Docker Compose

### Agent Capabilities

1. **Manager Agent**: Orchestrates workflow and routes to appropriate specialists
2. **Inquiry Agent**: Handles information requests using RAG
3. **Booking Agent**: Manages appointment scheduling and modifications
4. **General Agent**: Handles general conversations and FAQs
5. **SQL Agent**: Executes database queries for real-time data retrieval

## 🛠️ Tech Stack

### Core Framework
- **FastAPI** - Modern, fast web framework for building APIs
- **LangGraph** - Multi-agent workflow orchestration
- **LangChain** - LLM application framework
- **Pydantic** - Data validation and settings management

### AI/ML
- **DeepInfra** - LLM and embedding model hosting
- **Qdrant** - Vector database for semantic search
- **XGBoost** - Reranking model for search optimization
- **MLflow** - Model lifecycle management

### Data & Storage
- **PostgreSQL** - Primary relational database
- **MinIO** - S3-compatible object storage for ML artifacts
- **Qdrant** - Vector database for embeddings

### Frontend
- **Gradio** - Interactive web UI for chat interface

### Infrastructure
- **Docker** & **Docker Compose** - Containerization and orchestration
- **uv** - Fast Python package manager

## 🏗️ Architecture

Zenith AI follows a microservices architecture with clear separation of concerns:

```
┌─────────────┐
│   Gradio    │  Frontend UI (Port 7860)
│   Frontend  │
└──────┬──────┘
       │ HTTP
┌──────▼──────┐
│   FastAPI   │  REST API (Port 8000)
│   Backend   │
└──────┬──────┘
       │
┌──────▼──────────────────────────────────┐
│         LangGraph Workflow              │
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │ Manager  │→ │ Inquiry  │  │Booking ││
│  │  Agent   │  │  Agent   │  │ Agent  ││
│  └──────────┘  └──────────┘  └────────┘│
└──────┬──────────────────────────────────┘
       │
┌──────▼──────┬──────────┬──────────┬──────┐
│   Qdrant   │PostgreSQL│  MLflow  │DeepInfra
│  (Vector)  │   (SQL)  │ (Models) │  (LLM)
└────────────┴──────────┴──────────┴──────┘
```

For detailed architecture documentation, see [ARCHITECTURE.md](./ARCHITECTURE.md).

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10, 3.11, or 3.12
- **Docker** and **Docker Compose**
- **uv** package manager (recommended) or pip
- **DeepInfra API Token** ([Get one here](https://deepinfra.com))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ssabrut/zenith-ai.git
   cd zenith-ai
   ```

2. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv pip install -r requirements.txt
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start services with Docker Compose**
   ```bash
   make deploy
   # Or manually:
   docker compose up -d
   ```

5. **Access the application**
   - **Gradio UI**: http://localhost:7860
   - **API Docs**: http://localhost:8000/docs
   - **MLflow UI**: http://localhost:5050
   - **Qdrant Dashboard**: http://localhost:6333/dashboard
   - **MinIO Console**: http://localhost:9001

### Local Development Setup

For local development without Docker:

1. **Set environment variables**
   ```bash
   export IS_DOCKER=false
   # ... other environment variables
   ```

2. **Start infrastructure services**
   ```bash
   docker compose up -d qdrant db mlflow_db s3
   ```

3. **Run the FastAPI server**
   ```bash
   uvicorn core.main:app --reload --port 8000
   ```

4. **Run the Gradio frontend** (in another terminal)
   ```bash
   python frontend/main.py
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# Application
IS_DOCKER=false
SERVICE_NAME=front-desk-agent
APP_VERSION=0.1.0

# DeepInfra (LLM & Embeddings)
DEEPINFRA_API_TOKEN=your_token_here
DEEPINFRA_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
DEEPINFRA_CHAT_MODEL=your_chat_model

# Qdrant (Vector Database)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=zenith_collection

# PostgreSQL
POSTGRES_USER=zenith_user
POSTGRES_PASSWORD=zenith_password
POSTGRES_DB=zenith_db

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5050
MLFLOW_DB_USER=mlflow_user
MLFLOW_DB_PASSWORD=mlflow_password
MLFLOW_DB_NAME=mlflow_db

# MinIO (S3-compatible)
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MLFLOW_S3_ENDPOINT_URL=http://localhost:9002
MLFLOW_S3_IGNORE_TLS=true

# MCP Server (Optional)
MCP_SERVER_URL=http://localhost:8001/sse
```

## 📚 API Documentation

### Endpoints

#### Health Check
```http
GET /api/v1/health
```

#### Chat Endpoint
```http
POST /api/v1/chat
Content-Type: application/json

{
  "query": "What is the price of facial treatment?",
  "thread_id": "unique-thread-id"
}
```

**Response**: Streaming text/plain

### Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 💻 Development

### Project Structure

```
zenith-ai/
├── core/                    # Core application logic
│   ├── main.py             # FastAPI application entry point
│   ├── config.py           # Configuration management
│   ├── routers/            # API route handlers
│   │   ├── chat.py        # Chat endpoint
│   │   └── health.py      # Health check endpoint
│   ├── schemas/            # Pydantic models
│   └── services/           # External service clients
│       ├── deepinfra/      # LLM service integration
│       ├── qdrant/         # Vector database client
│       └── mlflow/         # MLflow client
├── graph/                   # LangGraph workflow
│   ├── workflow.py         # Graph definition
│   ├── state.py            # State management
│   ├── agent/              # Agent implementations
│   ├── node/               # Graph nodes
│   └── tools/               # Agent tools
├── frontend/                # Gradio UI
│   └── main.py             # Gradio application
├── notebooks/               # Jupyter notebooks for development
├── data/                    # Data storage (gitignored)
├── docker/                  # Dockerfiles
├── scripts/                 # Utility scripts
├── docker-compose.yml       # Service orchestration
├── pyproject.toml          # Project dependencies
└── requirements.txt        # Compiled dependencies
```

### Code Quality

The project uses:
- **Black** for code formatting
- **isort** for import sorting
- **Pydantic** for type validation

Format code:
```bash
black .
isort .
```

### Running Tests

```bash
# Add your test commands here
pytest tests/
```

### Adding New Agents

1. Create agent class in `graph/agent/`
2. Create node class in `graph/node/`
3. Add node to workflow in `graph/workflow.py`
4. Update routing logic in manager agent

### Database Migrations

```bash
# Using Alembic (if configured)
alembic upgrade head
```

## 📊 Monitoring & Observability

- **MLflow**: Track model versions, metrics, and artifacts
- **Health Checks**: Built-in health endpoints for monitoring
- **Logging**: Structured logging with Loguru

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write docstrings for all public functions
- Add type hints to function signatures
- Update documentation for new features
- Write tests for new functionality

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the LLM framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Qdrant](https://qdrant.tech/) for vector search capabilities

## 📧 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

<div align="center">

**Built with ❤️ using LangGraph, FastAPI, and modern AI technologies**

⭐ Star this repo if you find it helpful!

</div>
