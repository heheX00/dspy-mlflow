FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--dev", "--allowed-hosts", "*"]