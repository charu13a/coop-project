# GUI and API for Word Recommendation for crosswords
An interactive web-based GUI which allows you to get word suggestions for crosswords.

## Quick Start:
To start server:
1. ```virtualenv env```
2. ```source env/bin/activate```
3. ```pip install -r requirements.txt``` # requirements.txt contains all the python libraries required
4. ```cd crossword_recommender```
5. Download GoogleNews word embeddings from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) and place it in this folder.
5. ```python manage.py runserver```

Server running at localhost:8000 !

Two example puzzles included: examples/gandhi.txt, examples/sports_star.txt

## Directory structure:

(venv) charuagarwal@Charus-MacBook-Air django-apps % tree crossword_recommender

```
crossword_recommender
├── GoogleNews-vectors-negative300.bin # binary file for word embeddings
├── crossword_recommender
│   ├── settings.py # settings for the server
│   ├── urls.py # list routes URLs to views
│   └── wsgi.py
├── manage.py # to start the server
├── templates # templates common across the application (currently empty)
└── word_suggestions
    ├── admin.py
    ├── apps.py
    ├── migrations
    ├── model.py # algorithm for suggesting words
    ├── models.py # django file to connect to db (currently empty)
    ├── templates
    │   └── index.html # Main page UI for crossword recommendation
    ├── tests.py
    ├── urls.py
    └── views.py # intercepts requests from URL and has logic for returning response
```
