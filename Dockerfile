FROM python:3.7-slim

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . /app

EXPOSE 8080

ENTRYPOINT [ "python" ]

CMD [ "main.py" ]
