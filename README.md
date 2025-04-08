<<<<<<< HEAD
# vision-rag
## Requirements

- Python 3.8 or later

#### Install Dependencies

```bash
sudo apt update
sudo apt install libpq-dev gcc python3-dev
```
## Installation

### Install the required packages

```bash
$ pip install -r requirements.txt
```

### Setup the environment variables

```bash
$ cp .env.example .env
```

Set your environment variables in the `.env` file. Like `OPENAI_API_KEY` value.

## Run Docker Compose Services

```bash
$ cd docker
$ cp .env.example .env
```

- update `.env` with your credentials



```bash
$ cd docker
$ sudo docker compose up -d
```

## Run the FastAPI server

```bash
$ uvicorn main:app --reload --host 0.0.0.0 --port 5000
```
## Run the Interface

```bash
$ streamlit run app.py
```

