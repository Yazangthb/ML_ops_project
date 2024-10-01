# code/deployment/api/Dockerfile
FROM python:3.9

WORKDIR /app

COPY main.py /app/main.py
COPY ../../../models/iris_model.pkl /app/iris_model.pkl

RUN pip install fastapi uvicorn numpy scikit-learn pydantic

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
