# RGrid Machine Learning Challenge

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Welcome to the RGrid Machine Learning Challenge!

## Goal:

To test your ability to deal with text data, build a basic text classifier, and serve a model using an API framework (flask).

## Details:

In the `data` folder, you will find one `csv` file containing real text from medical trials scraped from ClinicalTrials.gov. Each row contains a `description` with text information about the trial and a corresponding `label` that identifies the disease or condition that it pertains to. There is an additional field `nctid` that describes the trial id that the row pertains to.

Provided with this repo is also a `main.py` file with a minimal [Flask](https://flask.palletsprojects.com/en/1.1.x/) demo. Once you have installed the `requirements.txt` in your python environment you will be able to run the main file by simply calling `python main.py` inside your directory. This should start the local server and you should be able to see `Hello World!` in your browser at `http://127.0.0.1:5000/`.

## The Task

Your task is to use the data to make a model capable of predicting a specific label given an unseen trial description.

You may assume that descriptions passed will always be relevant to at least one of the labels.

You should treat the `description` column as your input (X) and the `label` column as the output category (y) that you are trying to predict.

The `label` distribution is the following:

| `label`                       | Number of Examples |
| ----------------------------- | :----------------: |
| Dementia                      |        368         |
| ALS                           |        368         |
| Obsessive Compulsive Disorder |        358         |
| Scoliosis                     |        335         |
| Parkinson’s Disease           |        330         |

The `description` column contains plain text that explains the trial in a short description.

Your model should then be served through the small [Flask](https://flask.palletsprojects.com/en/1.1.x/) file provided. Feel free to extend it or split the file as needed.

### Before you pick a model/approach:

We suggest you take a look and explore the data. Look at the length of the `description` data and how you can handle inputs with variable lengths.

### Implementation Details

The preprocessing/algorithms/loss functions are yours to decide, as well as the separation between train/validation/test set. You do not necessarily have to use all the data provided either if you feel like some of it is irrelevant or not useful, as long as you can justify the choices that you make.

Specific design choices must be justified, whether that be qualitatively through plots or quantitatively through metrics. We want you to explain to us why you make the choices that you make, and to show us how good your solution is.

### Serving your solution through Flask

Once you have a model implemented which can predict the given output, we would like you to try to serve it in the API file provided (`main.py`).

This requires you to move your preprocessing to a function which you can call any input on.

You can then move your predict functionality to the `predict` function and return the predicted class.

You can test your API using the `test.py` file, just make sure you are running the server by calling `python main.py` in another terminal window.

## What we expect:

- Use of Python (3.8+)
- Clearly documented code or explanations with each function. **You need to be able to justify your design choices** - from data processing to algorithm decisions.

What we really care about is getting to know your thought process and decision making!
We are not going to penalize you for accuracy, but we would encourage you to be ready to discuss the strengths and weaknesses of your approach, and possible ways to improve your predictions.

You can use whatever external software libraries you think are appropriate. Numpy/scikit-learn/spacy are encouraged!

Your solution must be able to run and respond to requests (it can take as long to calculate as you want). You can imagine it as a micro-service that could be run independently on a server. Additional notebooks, analysis or plots to accompany your model will be very welcomed!

#### To Submit

You can email us a github/gitlab link to your solution (preferred) or a zipped file with your code and explanations.
We look forward to your solution 🙂
