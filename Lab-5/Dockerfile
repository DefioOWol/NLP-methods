FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

COPY requirements.txt torch-cpu-req.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r torch-cpu-req.txt

CMD ["sleep", "infinity"]
