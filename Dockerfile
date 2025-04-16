FROM python:latest

WORKDIR /face_app
ARG PORT=80
ENV PORT=$PORT
# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .
COPY app/haarcascade_frontalface_default.xml .
COPY app/verified_gender_model.pkl .
EXPOSE $PORT

RUN pip install  -r requirements.txt

CMD ["python", "main.py"]

