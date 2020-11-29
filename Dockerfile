FROM python:3.8-slim

RUN mkdir -p /app/
COPY requirements.txt ./app

RUN pip install --no-cache-dir -r ./app/requirements.txt

RUN pip install ipython==7.17.0
RUN pip install tensorflow-datasets==3.2.1

COPY ./auxiliary_lib/tensorflow_examples /usr/local/lib/python3.8/site-packages/tensorflow_examples

COPY ./app ./app

WORKDIR /app/

RUN chmod +x ./run.sh

ENTRYPOINT ["./run.sh"]
# ENTRYPOINT ["bash"]