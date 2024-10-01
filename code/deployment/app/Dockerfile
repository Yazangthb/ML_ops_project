# code/deployment/app/Dockerfile
FROM python:3.9

WORKDIR /app

COPY app.py /app/app.py

RUN pip install streamlit requests

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
