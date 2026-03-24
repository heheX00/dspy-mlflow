# 1. Use the official lightweight Python image
FROM python:3.12.12-slim

# 2. Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /code

# 4. Copy ONLY the requirements file first to leverage Docker cache
COPY ./requirements.txt /code/requirements.txt

# 5. Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 6. Copy the rest of your application code
COPY . .
# 7. Expose the port the app runs on (optional but good practice for documentation)
EXPOSE 8000

# 8. Define the command to start the Uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]