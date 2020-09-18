# Install URLPhish

## Docker

Build a Docker image from Dockerfile as that already have the correct dependencies to run the code.

### RUN on shell

```shell
$ cd /path/to/URLPhishing/DOCKERFILE/
$ docker build -t urlphish:latest .
```

and then run the container:

```shell
$ nvidia-docker run --tty --interactive --userns=host --volume /path/to/URLPhishing:/home/<your-name>/work --name <container-name> urlphish:latest  /bin/bash

```

Finally, install the URLPhish:

```shell
$ python setup.py install

```

### Train the model from scratch

```shell
python ./URLPhishing/src/general_pipline.py -l url_phishing_log.app -s False -t ./path/to/train/dataset.csv/ -v ./path/to/validation/dataset.csv/ -m model
```

### Train the model using the already trained vocabulary

> Note that the files vocab_label.pkl, vocab_sen.pkl... will be loaded. Make sure you run the code on the root directory.

```shell
python ./URLPhishing/src/general_pipline.py -l url_phishing_log.app -s True -t ./path/to/train/dataset.csv/ -v ./path/to/validation/dataset.csv/ -m model
```

### To classify a single URL

```shell
python ./URLPhishing/src/classify.py -u 'www.facebook.com' -i ./model -s vocab_sen.pkl -l vocab_label.pkl -r log_report.app

```

# Contributing

1. Fork this repository
2. Make a new branch: `git checkout -b new-branch-name`
3. Commit the modifications: `git commit -m 'changes description'`
4. Push for the remote repository: `git push -u origin URL`

