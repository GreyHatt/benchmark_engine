# benchmark_engine


## Repository Structure 

```
benchmark_engine/
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.spark
│   └── docker-compose.yml
├── src/
│   ├── api/                  
│   │   ├── main.py
│   │   ├── routers/
│   │   └── models.py
│   ├── benchmark/           
│   │   ├── spark_engine.py
│   │   ├── duckdb_engine.py
│   │   └── hybrid_engine.py
│   ├── data/                
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   └── loader.py
│   └── visualization/       
│       ├── app.py
│       └── utils.py
├── tests/                   
├── requirements/            
│   ├── base.txt
│   ├── api.txt
│   └── visualization.txt
└── README.md
```