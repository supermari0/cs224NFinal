This file gives instructions for how to install the appropriate dependencies
for and run a classifier which predicts whether a given Shakespeare speech was
given by a male or female character.

How to Install
- Install NLTK using the directions here: http://www.nltk.org/install.html
- Install scikit-learn using "sudo pip install scikit-learn" or the directions
  here: http://scikit-learn.org/stable/install.html
- Open a Python interpreter and type the following at the prompt:

>>> import nltk
>>> nltk.download('maxent_treebank_pos_tagger')

The classifier should now be ready to prompt. The classifier has only been
tested with Python 2.7.5.

How to Run
Type "python classify.py -h" in the main project directory to get a list of
options for the classifier. You shouldn't need to use any of the other files to
run the classifier.
