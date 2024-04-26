FROM python:3.9-slim-bullseye
WORKDIR /app
COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt
CMD ["gunicorn", "--bind", "0.0.0.0:5555", "main:application"]