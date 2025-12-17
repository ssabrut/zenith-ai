# Dockerfile.frontend
FROM python:3.10-slim

# 1. Set working directory
WORKDIR /app

# 2. Install Dependencies
# We install the same requirements because Gradio imports the core graph
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the entire project
# We need 'core/' because app_frontend.py imports 'graph.workflow'
COPY . .

# 4. Set Environment Variables
# Crucial for Gradio to work inside Docker
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
ENV PYTHONPATH=/app

# 5. Expose the port
EXPOSE 7860

# 6. Run the App
CMD ["python", "frontend/main.py"]