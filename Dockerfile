FROM python:3.8-slim-buster 
# Assuming you have a FROM line like: FROM python:3.8-slim
WORKDIR /app
COPY . /app

# 1. Install AWS CLI via pip instead of apt
RUN pip install --no-cache-dir awscli

# 2. Install your other requirements
RUN pip install -r requirements.txt

CMD [ "python3", "app.py" ]