# Official Jupyter Notebook image as the base
FROM jupyter/base-notebook:latest

# Set the working directory inside the container
WORKDIR /app

# Install necessary system dependencies
USER root
RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    gcc \
    g++ \
    make \
    p7zip-full

# Switch back to the jovyan user, typical for Jupyter Docker images
USER jovyan

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Jupyter Notebook files in the root directory to the dockerimage
COPY . .

# Expose the port that Jupyter Notebook runs on
EXPOSE 8888

# Command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
