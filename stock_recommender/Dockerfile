# Use a Python 3.9 slim-buster image as the base
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the Python dependencies, including pytest
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files into the container
COPY . .

# Command to run the application by default
# This can be overridden at runtime. For Cloud Run Jobs, arguments are specified in job config.
# For testing purposes, we can override CMD or run pytest manually after building the image.
CMD ["python", "src/agent.py", "AAPL", "--period", "1y"]

# Optional: Add a test command as a separate entrypoint or during build
# For running tests, you would typically build the image and then run a separate container:
# docker run --rm <image_name> pytest
