FROM python:3.8

COPY ./requirements.txt /home/automl/
WORKDIR /home/automl/

RUN python3 -m pip install -r requirements.txt

COPY automl /home/automl/

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]