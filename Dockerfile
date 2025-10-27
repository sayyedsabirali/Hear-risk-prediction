# ✅ ALTERNATIVE OPTION:
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

COPY . /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]