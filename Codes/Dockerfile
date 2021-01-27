FROM tensorflow/tensorflow

RUN cd /home && mkdir lucas
RUN cd /home/lucas
RUN mkdir DataLoad && mkdir Model && mkdir particles && mkdir uhcs && mkdir RunData
COPY DataLoad/ /home/lucas/DataLoad/
COPY Model/ /home/lucas/Model/
COPY particles/ /home/lucas/particles/
COPY uhcs/ /home/lucas/uhcs/
COPY training_uhcs_pixelnet.py /home/lucas/
COPY requirements.txt /home/lucas/
COPY settings.yaml /home/lucas/
COPY client_secrets.json /home/lucas/
COPY credentials.json /home/lucas/

RUN apt-get update
RUN pip install -r /home/lucas/requirements.txt 

ENTRYPOINT [ "/bin/bash" ]


