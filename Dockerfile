FROM python:3.6-stretch
LABEL maintainer="bourgonje@uni.potsdam.de"

RUN apt-get -y update &&\
    apt-get upgrade -y &&\
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y curl && \
    apt-get install -y expect && \
    apt-get install -y python3-dev &&\
    apt-get update -y

ADD requirements.txt .
RUN pip3 install -r requirements.txt
RUN python3 -m nltk.downloader punkt

RUN mkdir gsdp
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
ADD stanford-parser-full-2018-10-17 /gsdp/stanford-parser
ADD bert-base-german-cased_tf_version /gsdp/bert-base-german-tf-version
ADD pcc2.2 /gsdp/pcc2.2
ADD dimlex /gsdp/dimlex
ADD sleep.sh /gsdp/.
ADD docker_start.sh .

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV CLASSPATH=/gsdp/stanford-parser/stanford-parser.jar

EXPOSE 5000

ENTRYPOINT ["./docker_start.sh"]
#CMD ["/bin/bash"]
