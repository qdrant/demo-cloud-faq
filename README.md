# Question Answering tutorial with Quaterion

## Intro

Similarity learning can be handy in a large amount of domains and tasks.
It can be used in CV, NLP, recommendation systems, etc.

You may need to look at problems you are already used to from a different angle in order to solve them with similarity learning.
At first, it may be unusual and to mitigate "mind-shift" we present our series of [tutorials](https://quaterion.qdrant.tech/tutorials/tutorials.html).

This tutorial covers one of NLP problems - Q&A. 
Here we train a model capable of answering to questions from F.A.Q. pages of popular cloud providers. 

The whole tutorial is available [here](https://quaterion.qdrant.tech/tutorials/nlp_tutorial.html)

## Dependencies

We use [poetry](https://python-poetry.org/) to manage our dependencies.

All packages required are listed in `pyproject.toml`.

If you have never used `poetry` earlier, we've collected a bunch of commands to make it seamless in `install_dependencies.sh`.
You can install all the dependencies with only one command: 
```shell
./install_dependencies.sh
```

Make this file executable with `chmod +x install_dependencies.sh` before usage.

## Dataset

In this tutorial we use our own [dataset](https://github.com/qdrant/dataset-cloud-platform-faq).

You can download it with 
```shell
./download_data.sh
```

Make this file executable with `chmod +x download_data.sh` before usage.

It will fetch `cloud_faq_dataset.jsonl` file from our Google Storage and make a train-val split.








