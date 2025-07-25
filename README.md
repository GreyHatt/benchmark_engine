# TPC-H Benchmark Engine

A high-performance benchmarking platform for comparing query execution engines using the TPC-H benchmark. This project provides a unified interface to run and compare queries across different database engines with detailed performance metrics and visualizations.

## Features

- **Multiple Query Engines**: Supports Spark, and DuckDB execution
- **TPC-H Benchmark**: Standardized TPC-H queries for fair comparison
- **RESTful API**: FastAPI-based backend for programmatic access
- **Interactive Dashboard**: Streamlit-based UI for visualization and analysis
- **Containerized Deployment**: Ready for Docker and Kubernetes
- **CI/CD**: Automated testing and deployment with GitHub Actions

## Project Structure

```
benchmark_engine/
├── .github/workflows/      # CI/CD workflows
│   └── deploy.yml          # Deployment pipeline
├── infra/                  # Infrastructure as Code
│   └── k8s-deployment.yml  # Kubernetes deployment config
├── src/
│   ├── api/                # FastAPI application
│   │   └── main.py         # API endpoints and request handlers
│   ├── data/               # Data management
│   │   ├── generator.py    # TPC-H data generation
│   │   └── loader.py       # Data loading utilities
│   ├── query/              # Query execution
│   │   ├── base.py         # Base query executor
│   │   ├── duckdb_executor.py  # DuckDB implementation
│   │   ├── spark_executor.py   # Spark implementation
│   │   ├── factory.py      # Query executor factory
│   │   ├── metrics.py      # Performance metrics collection
│   │   └── tpch_queries.py # TPC-H query definitions
│   └── streamlit_app.py    # Interactive dashboard
├── docker-compose.yml      # Local development setup
├── requirements.txt        # Python dependencies   
└── Dockerfile              # Production container definition
```

## Getting Started

### Prerequisites

- Python 3.9+
- Java 17 (for Spark)
- Docker and Docker Compose (for containerized deployment)
- Kubernetes cluster (for production deployment)

### Installation

#### Clone the repository:
   ```bash
   git clone git@github.com:GreyHatt/benchmark_engine.git
   cd benchmark_engine
   ```

### Using Docker Compose

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000` with interactive documentation at `/docs`.

Streamlit dashbaord will be available at `http://localhost:8501`.


## API Endpoints

- `GET /queries` - List all available TPC-H queries
- `GET /engines` - List available query engines
- `GET /data/status` - Check TPC-H data status
- `POST /data/generate` - Generate TPC-H test data
- `POST /benchmark` - Run a benchmark
- `GET /benchmark/{benchmark_id}` - Get benchmark results


### GitHub Actions

CI/CD is configured to automatically build and deploy the application on pushes to the `main` branch.