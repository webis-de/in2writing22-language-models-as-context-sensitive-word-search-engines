# Language Models as Context-sensitive Word Search Engines

This repository is the official implementation of [Language Models as Context-sensitive Word Search Engines](TODO) at the in2writing workshop at ACL22. 

```
@inproceedings{wiegmann:2022,
    title =     "Language Models as Context-sensitive Word Search Engines",
    author =    "Wiegmann, Matti and V{\"{o}}lske, Michael and Potthast, Martin and Stein, Benno",
    booktitle = "Proceedings of the 1st Workshop on Intelligent and Interactive Writing Assistants (In2Writing 2022)",
    month =     may,
    year =      "2022",
    address =   "Online",
    publisher = "Association for Computational Linguistics",
}
```

## Requirements

To install requirements:

```setup
~$ python3 -m venv venv 
~$ source venv/bin/activate
(venv) ~$ pip install wheel
(venv) ~$ pip install .
(venv) ~$ python -m spacy download en_core_web_trf
```

To start the cli and inspect the help:

```bash
(venv) ~$ main
```

## Training

To train the models in the paper, run these commands:

```train
(venv) ~$ main train --output_path "/save/model/here"
    --training_file "/path/to/train.ndjson" \
    --validation_file "/path/to/validation.ndjson" \   
    --model_name "distilbert-base-uncased" \  # or facebook/bart-base for bart
    --strategy "mlm"  # or "s2sranked" for bart 
```

The cli offers hyperparameter options, but the defaults reflect the published method. 

## Evaluation

The test dataset for this work can be found on [zenodo](https://doi.org/10.5281/zenodo.6425595)
To evaluate the models on the test datasets, generate the predictions first: 

```eval
(venv) ~$ main test --input_file "/path/to/test.ndjson" \   
    --model_name "/path/to/trained/model" \ 
    --task "mlm"  # for dBERT or "s2sranked" for bart 
```

Then, run the evaluation scripts to generate the output results. 

```
(venv) ~$ main evaluate -t /path/to/test.ndjson
```

## Results

Our model achieves the following performance:

| Model     | wt3      | wt3      | wt5      | wt5      | cl3      | cl3      | cl5      | cl5 |
| --------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
|           | RA       | OT       | RA       | OT       | RA       | OT       | RA       | OT |
| Netspeak  | 0.33     | --       | 0.46     | --       | 0.10     | --       | 0.22     | -- | 
| dBERT     | 0.15     | 0.14     | 0.33     | 0.28     | 0.06     | 0.06     | **0.17** | **0.15** |
| dBERT(ft) | **0.30** | **0.29** | 0.42     | **0.35** | 0.05     | 0.05     | 0.10     | 0.08 |
| BART      | 0.19     | 0.18     | 0.37     | 0.31     | 0.05     | 0.05     | 0.15     | 0.12 |
| BART(ft)  | 0.29     | 0.28     | **0.43** | 0.34     | **0.07** | **0.07** | **0.17** | 0.12 |