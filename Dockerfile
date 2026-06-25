# Use an official lightweight Python image
FROM python:3.10-slim

# Set up a working directory inside the container
WORKDIR /code

# Copy your requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all your project files into the container
COPY . .

# Expose port 7860 (Hugging Face's default port)
EXPOSE 7860

# Start your FastAPI app using uvicorn on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]