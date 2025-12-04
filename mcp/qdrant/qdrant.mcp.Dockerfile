FROM python:3.10-slim

WORKDIR /app

# Install system deps (if needed for xgboost/pandas)
RUN apt-get update && apt-get install -y libgomp1

COPY ../../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port (FastMCP uses 8000 by default, but we can change it)
EXPOSE 8080

# Run with SSE transport
CMD ["python", "server.py"]