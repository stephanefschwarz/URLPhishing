# FROM tensorflow/tensorflow:latest-gpu-py3

FROM pytorch/pytorch

RUN apt update
RUN apt install nano
RUN apt-get install -y git 
RUN apt-get install -y wget 

RUN apt update

RUN apt install whois
RUN apt install netbase

RUN pip install --upgrade pip 
RUN pip install pandas
RUN pip install numpy
RUN pip install tldextract
RUN pip install tqdm
RUN pip install whois
RUN pip install scikit-learn
RUN pip install matplotlib
RUN pip install seaborn
RUN conda install -c anaconda pytorch-gpu 
# RUN pip install torch


