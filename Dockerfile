FROM python:3.8

WORKDIR /usr/src/app

COPY . ./

RUN pip install pipenv
RUN pipenv install
RUN ls -la

RUN pipenv run python -m unittest discover inversion/
CMD sh