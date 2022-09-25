FROM python:3.9.14-bullseye
COPY ./ /app
WORKDIR /app
RUN ls -a
RUN pip3 install -r requirements.txt
CMD ["gunicorn", "--bind", "0.0.0.0:5555", "main:application"]
