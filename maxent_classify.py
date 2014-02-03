import collections
import math
import nltk
import pickle

TRAIN_DATA_FILE = 'data/proc_train'
DEV_DATA_FILE = 'data/proc_dev'
TEST_DATA_FILE = 'data/proc_test'

COMMON_TOKENS_FILE = 'data/top_tokens'

def common_token_feature_dict(common_tokens_dict, token_tuples):
    """ common_tokens_dict is the dictionary whose keys are the most common
    tokens as defined in the common_token_features function and whose values
    are initialized to 0. token_tuples is a list of (token, part-of-speech)
    tuples. The function returns a dictionary whose keys are the keys of
    common_tokens_dict and whose values are the frequency of occurrence of the
    key token in the text (1 + ln(count/n_tokens) if count != 0, 0 otherwise). """


    for (token, pos) in token_tuples:
        if token in common_tokens_dict:
            common_tokens_dict[token] = 1

    return common_tokens_dict
    #for (token, pos) in token_tuples:
    #    if token in common_tokens_dict:
    #        common_tokens_dict[token] += 1

    #n_tokens = len(token_tuples)
    #for token, count in common_tokens_dict.items():
    #    if count != 0:
    #        common_tokens_dict[token] = 1 + math.log( (count + 1) / (float(n_tokens
    #            + len(common_tokens_dict))))
    #    else:
    #        common_tokens_dict[token] = 1 + math.log( 1 / (float(n_tokens +
    #             len(common_tokens_dict)))) 

    #return common_tokens_dict

def common_token_features(proc_data, label):
    """ Returns a list of (feature dictionary, label) pairs where the features
    are the frequency of occurrence of the most common tokens. The most common
    tokens are defined as the intersection of the top 300 tokens said by male
    characters and top 300 tokens said by female characters in the training
    data. These are precomputed. """

    common_tokens_pickled = open(COMMON_TOKENS_FILE, 'rb')
    common_tokens_set = pickle.load(common_tokens_pickled)
    common_tokens_pickled.close()

    features = []

    for speech_tuple in proc_data:
        common_tokens_dict = dict.fromkeys(common_tokens_set, 0)

        token_tuples = speech_tuple[0]
        if label:
            gender_tag = speech_tuple[1]

        if label:
            features.append((common_token_feature_dict(common_tokens_dict, token_tuples),
                gender_tag))
        else:
            features.append(common_token_feature_dict(common_tokens_dict,
                token_tuples))
    return features

def extract_features(proc_data, label=False):
    features = []
    features += common_token_features(proc_data, label)
    return features

def extract_labels(proc_data):
    labels = []
    for speech_tuple in proc_data:
        labels += speech_tuple[1]
    return labels

def get_training_set():
    train_pickled = open(TRAIN_DATA_FILE, 'rb')
    train_proc_data = pickle.load(train_pickled)
    train_pickled.close()
   
    featureset = extract_features(train_proc_data, label=True)

    return featureset

def get_dev_set():
    """ Returns a tuple whose first element is the unlabeled list of feature
    dicts ready for classification and whose second element is the list of
    gold labels. """
    dev_pickled = open(TEST_DATA_FILE, 'rb')
    dev_proc_data = pickle.load(dev_pickled)
    dev_pickled.close()

    featureset = extract_features(dev_proc_data)
    labels = extract_labels(dev_proc_data)
    print labels
    return (featureset, labels)


def classify():
    training_set = get_training_set()
    
    # Set algorithm to GIS because of bug in scipy (missing maxentropy module).
    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]

    classifier = nltk.classify.MaxentClassifier.train(training_set, algorithm)

    dev_set = get_dev_set()
    labels = classifier.batch_classify(dev_set[0])
    print labels

    correct = 0
    for i in range(len(labels)):
        if labels[i] == dev_set[1][i]:
            correct += 1
    print(float(correct)/len(labels))    

if __name__ == '__main__':
    classify()
