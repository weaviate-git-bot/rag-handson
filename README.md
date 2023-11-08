# RAG HandsOn

## Pré-requisitos

- Docker daemon/cli client
- Miniconda
- Git
- VSCode

### Conda

1. Criar ambiente virtual
```shell
conda create --name ml -c conda-forge python=3.11
conda activate ml
```

2. Instalar dependências
```shell
conda install -c conda-forge jupyter pandas numpy sentence-transformers tensorflow 
pip install -U ibm-generative-ai "ibm-generative-ai[langchain]" pypdf readchar weaviate-client python-dotenv langchain huggingface torch
```

3. Testar dependências
```shell
curl -LJO https://github.com/vanildo/rag-handson/raw/main/configTest.py
python ./configTest.py
```

### Weaviate

- Site: <https://weaviate.io>
- Github: <https://github.com/weaviate/weaviate>
- Baixar o arquivo ```docker-compose.yml```

```shell
curl -LJO https://github.com/vanildo/rag-handson/raw/main/docker-compose.yml
```

- Comando para rodar local:

```shell
docker compose up -d
```

para rodar o docker compose com suporte a persistência, tem que cria a pasta __weaviate_data__.

```shell
mkdir weaviate_data
```

## Elasticsearch

- <https://www.elastic.co/guide/en/elasticsearch/reference/current/date.html>
