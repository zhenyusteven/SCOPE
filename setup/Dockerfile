FROM python:3.10-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install pandas pyarrow

WORKDIR /app

COPY clone_repos.py /app/clone_repos.py

CMD ["python", "clone_repos.py"]
