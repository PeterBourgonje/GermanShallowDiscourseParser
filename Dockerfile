FROM python:3.6-stretch
LABEL maintainer="bourgonje@uni.potsdam.de"

RUN apt-get -y update &&\
    apt-get upgrade -y &&\
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y curl && \
    apt-get install -y expect && \
    apt-get install -y git && \
    apt-get install -y python3-dev &&\
    apt-get update -y

ADD requirements.txt .
RUN pip3 install -r requirements.txt
RUN python3 -m nltk.downloader punkt

RUN mkdir gsdp

RUN wget https://nlp.stanford.edu/software/stanford-parser-full-2018-10-17.zip
RUN unzip stanford-parser-full-2018-10-17.zip -d /gsdp/stanford-parser && rm stanford-parser-full-2018-10-17.zip

RUN wget https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/tensorflow/bert-base-german-cased.zip
RUN unzip bert-base-german-cased.zip -d /gsdp/bert-base-german-tf-version && rm bert-base-german-cased.zip
RUN mv /gsdp/bert-base-german-tf-version/bert-base-german-cased.data-00000-of-00001 /gsdp/bert-base-german-tf-version/bert_model.ckpt.data-00000-of-00001
RUN mv /gsdp/bert-base-german-tf-version/bert-base-german-cased.index /gsdp/bert-base-german-tf-version/bert_model.ckpt.index
RUN mv /gsdp/bert-base-german-tf-version/bert-base-german-cased.meta /gsdp/bert-base-german-tf-version/bert_model.ckpt.meta


ADD docker_config.ini /gsdp/config.ini
ADD ConnectiveClassifier.py /gsdp/.
ADD DimLexParser.py /gsdp/.
ADD ExplicitArgumentExtractor.py /gsdp/.
ADD ExplicitSenseClassifier.py /gsdp/.
ADD ImplicitSenseClassifier.py /gsdp/.
ADD Parser.py /gsdp/.
ADD PCCParser.py /gsdp/.
ADD utils.py /gsdp/.

ADD bert_client_encodings.pickle /gsdp/.
ADD pcc_memorymap.pickle /gsdp/.
ADD sleep.sh /gsdp/.
ADD docker_start.sh .

RUN git clone https://github.com/discourse-lab/dimlex /gsdp/dimlex
RUN git clone https://github.com/PeterBourgonje/pcc2.2 /gsdp/pcc2.2


ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV CLASSPATH=/gsdp/stanford-parser/stanford-parser.jar

EXPOSE 5000

ENTRYPOINT ["./docker_start.sh"]
#CMD ["/bin/bash"]


