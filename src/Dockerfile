FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN apt-get update -y
RUN python -m pip install --upgrade pip
RUN pip install --upgrade wheel
RUN pip install --upgrade setuptools

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

CMD ["python", "seed.py"]