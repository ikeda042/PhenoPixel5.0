# Use the base image you are currently using
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY ./app/requirements.txt /app/requirements.txt

# Install dependencies including libglib2.0-0 for libgthread-2.0.so.0
# and other necessary libraries for running Chrome and ChromeDriver
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 \
    libnss3 libx11-xcb1 libxcomposite1 libxcursor1 libxdamage1 libxi6 libxtst6 \
    libxrandr2 libxss1 libasound2 libatk1.0-0 libatk-bridge2.0-0 libgbm1 libgtk-3-0 \
    wget unzip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    # Download ChromeDriver (adjust the version if needed)
    wget -O /tmp/chromedriver.zip https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip && \
    unzip /tmp/chromedriver.zip -d /usr/local/bin/ && \
    rm /tmp/chromedriver.zip

# Copy the rest of the application code
COPY ./app /app

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]