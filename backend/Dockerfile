# Use the base image you are currently using
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY ./app/requirements.txt /app/requirements.txt

# Install dependencies including libglib2.0-0 for libgthread-2.0.so.0
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 wget unzip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Install Google Chrome
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y ./google-chrome-stable_current_amd64.deb && \
    rm google-chrome-stable_current_amd64.deb

# Install ChromeDriver
RUN CHROME_VERSION=$(google-chrome --version | grep -oP '\d+\.\d+\.\d+') && \
    wget https://chromedriver.storage.googleapis.com/$CHROME_VERSION/chromedriver_linux64.zip && \
    unzip chromedriver_linux64.zip && \
    mv chromedriver /usr/local/bin/ && \
    rm chromedriver_linux64.zip

# Copy the rest of the application code
COPY ./app /app

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]