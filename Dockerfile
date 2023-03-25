FROM python:3.10.5
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 3000
CMD python ./app.py