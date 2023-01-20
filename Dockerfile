FROM python:3.9.16-slim-buster
WORKDIR /app
COPY . /app
EXPOSE 8501
RUN apt update -y\
    && pip install -U pip \
    && pip install -r requirements.txt
ENTRYPOINT [ "streamlit", "run", "app.py" ]