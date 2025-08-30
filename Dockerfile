FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your app code
COPY . /app
WORKDIR /app

CMD ["uvicorn", "detect:app", "--host", "0.0.0.0", "--port", "10000"]
