# Use the base image you are currently using
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY ./app/requirements.txt /app/requirements.txt

# Install dependencies including libglib2.0-0 for libgthread-2.0.so.0
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code
COPY ./app /app

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]