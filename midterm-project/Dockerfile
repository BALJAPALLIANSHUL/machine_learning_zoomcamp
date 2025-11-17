# 1. Use a standard, slim Python base image
FROM python:3.12-slim-bookworm

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install 'uv' using pip (this is a simple way to get it)
RUN pip install uv

# 4. Copy *only* the requirements file first.
# This is a Docker best-practice. It allows Docker to cache the
# installed packages (Step 5) unless this file changes.
COPY requirements.txt requirements.txt

# 5. Create the virtual environment AND install dependencies
# We create the .venv folder first
RUN uv venv
# Now we run 'sync' - uv is smart enough to find and use the .venv
RUN uv pip sync requirements.txt

# 6. Copy the rest of your application code and the model
# This copies your API script and the trained model .pkl file
COPY predict.py .
COPY paddy_yield_model_v1.pkl .

# 7. Expose the port your server will run on
EXPOSE 8000

# 8. Define the command to run your application
# We call 'uvicorn' directly from the .venv/bin folder we created.
CMD [".venv/bin/uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]