# Use an official Python 3.10 image from Docker Hub
FROM python:3.10-slim-buster

# Think of it like:
# “Build my container on top of an existing ready-made Python system”

# Set the working directory
WORKDIR /app

# ✅ SYSTEM DEPENDENCIES INSTALL KARO - YEH IMPORTANT HAI
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy your application code
COPY . /app

# Install the dependencies
RUN pip install -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 5000

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]