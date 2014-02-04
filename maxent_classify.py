import collections
import math
import nltk
import pickle

TRAIN_DATA_FILE = 'data/proc_train'
DEV_DATA_FILE = 'data/proc_dev'
TEST_DATA_FILE = 'data/proc_test'

COMMON_BIGRAMS_FILE = 'data/top_bigrams'
COMMON_TOKENS_FILE = 'data/top_tokens'
COMMON_TRIGRAMS_FILE = 'data/top_trigrams'

POS_FILE = 'data/pos_set'
POS_BIGRAMS_FILE = 'data/pos_bigrams'

N_ITERATIONS = 75

def pos_features(proc_data, label):
    # TODO add backoff, pos ngrams. use indicator vars instead of stats.
    # TODO parse and put POS from train including 'OTHER' in file to make this job easier
    # to handle
    """ Takes in process data and whether or not to label the features (whether
    or not the features should be added as part of a training set). Returns the
    a list of feature dictionaries where the features are the Laplace-smoothed 
    frequency ofccurrence of the parts of speech in the text. """

    pos_pickled = open(POS_FILE, 'rb')
    pos_set = pickle.load(pos_pickled)
    pos_pickled.close()
    features = []

    for speech_tuple in proc_data:
        pos_dict = dict.fromkeys(pos_set, 0)
        pos_dict['OTHER'] = 0
        token_tuples = speech_tuple[0]

        pos_features = pos_feature_dict(pos_dict, token_tuples)

        if label:
            gender_tag = speech_tuple[1]
            features.append((pos_features, gender_tag))
        else:
            features.append(pos_features)

    return features


def pos_feature_dict(pos_dict, token_tuples):

    for (token, pos) in token_tuples:
        if pos in pos_dict:
            pos_dict[pos] += 1
        else:
            pos_dict['OTHER'] += 1

    n_pos = len(token_tuples)
    for pos, count in pos_dict.items():
        pos_dict[pos] = math.log((count + 1) / (float(n_pos) + len(pos_dict)))

    return pos_dict

def common_token_feature_dict(common_tokens_dict, token_tuples):
    """ common_tokens_dict is the dictionary whose keys are the most common
    tokens as defined in the common_token_features function and whose values
    are initialized to 0. token_tuples is a list of (token, part-of-speech)
    tuples. The function returns a dictionary whose keys are the keys of
    common_tokens_dict and whose values are 1 if the token is in the speech, 0
    otherwise. """

    for (token, pos) in token_tuples:
        if token in common_tokens_dict:
            common_tokens_dict[token] = 1

    return common_tokens_dict

def common_token_features(proc_data, label):
    """ Returns a list of (feature dictionary, label) (label only if
    label==True) pairs where the features are the frequency of occurrence of
    the most common tokens. The most common tokens are defined as the
    intersection of the top 2K tokens said by male characters and top 2K
    tokens said by female characters in the training data. These are
    precomputed. """

    common_tokens_pickled = open(COMMON_TOKENS_FILE, 'rb')
    common_tokens_set = pickle.load(common_tokens_pickled)
    common_tokens_pickled.close()

    features = []

    for speech_tuple in proc_data:
        common_tokens_dict = dict.fromkeys(common_tokens_set, 0)


        token_tuples = speech_tuple[0]

        if label:
            gender_tag = speech_tuple[1]
            features.append((common_token_feature_dict(common_tokens_dict, token_tuples),
                gender_tag))
        else:
            features.append(common_token_feature_dict(common_tokens_dict,
                token_tuples))
    return features

def pos_bigram_features(proc_data, label):
    pos_bigrams_pickled = open(POS_BIGRAMS_FILE, 'rb')
    pos_bigrams_set = pickle.load(pos_bigrams_pickled)
    pos_bigrams_pickled.close()

    features = []

    for speech_tuple in proc_data:
        pos_bigrams_dict = dict.fromkeys(pos_bigrams_set, 0)

        token_tuples = speech_tuple[0]
        pos_bigram_features = pos_bigram_feature_dict(pos_bigrams_dict,
                token_tuples)

        if label:
            gender_tag = speech_tuple[1]
            features.append((pos_bigram_features, gender_tag))
        else:
            features.append(pos_bigram_features)

    return features

def pos_bigram_feature_dict(pos_bigrams_dict, token_tuples):
    for i in range(1, len(token_tuples)):
        pos = token_tuples[i][1]
        prev_pos = token_tuples[i-1][1]
        pos_bigram = prev_pos + '+' + pos
        if pos_bigram in pos_bigrams_dict:
            pos_bigrams_dict[pos_bigram] = 1
    return pos_bigrams_dict

def common_bigram_features(proc_data, label):
    common_bigrams_pickled = open(COMMON_BIGRAMS_FILE, 'rb')
    common_bigrams_set = pickle.load(common_bigrams_pickled)
    common_bigrams_pickled.close()

    features = []

    for speech_tuple in proc_data:
        common_bigrams_dict = dict.fromkeys(common_bigrams_set, 0)

        token_tuples = speech_tuple[0]
        bigram_features = common_bigram_feature_dict(common_bigrams_dict,
                token_tuples)
        
        if label:
            gender_tag = speech_tuple[1]
            features.append((bigram_features, gender_tag))
        else:
            features.append(bigram_features)

    return features

def common_bigram_feature_dict(common_bigrams_dict, token_tuples):
    for i in range(1, len(token_tuples)):
        token = token_tuples[i][0]
        prev_token = token_tuples[i-1][0]
        bigram = prev_token + '+' + token
        if bigram in common_bigrams_dict:
            common_bigrams_dict[bigram] = 1

    return common_bigrams_dict

def common_trigram_features(proc_data, label):
    common_trigrams_pickled = open(COMMON_TRIGRAMS_FILE, 'rb')
    common_trigrams_set = pickle.load(common_trigrams_pickled)
    common_trigrams_pickled.close()

    features = []

    for speech_tuple in proc_data:
        common_trigrams_dict = dict.fromkeys(common_trigrams_set, 0)

        token_tuples = speech_tuple[0]
        trigram_features = common_trigram_feature_dict(common_trigrams_dict,
                token_tuples)
        
        if label:
            gender_tag = speech_tuple[1]
            features.append((trigram_features, gender_tag))
        else:
            features.append(trigram_features)

    return features

def common_trigram_feature_dict(common_trigrams_dict, token_tuples):
    for i in range(2, len(token_tuples)):
        token = token_tuples[i][0]
        prev_token = token_tuples[i-1][0]
        prev2_token = token_tuples[i-2][0]
        trigram = prev2_token + '+' + prev_token + '+' + token
        if trigram in common_trigrams_dict:
            common_trigrams_dict[trigram] = 1

    return common_trigrams_dict


def extract_features(proc_data, label=False):
    #features = []
    #features += pos_features(proc_data, label)
    #common_trigram_feat = common_trigram_features(proc_data, label)
    features = common_bigram_features(proc_data, label)
    #pos_bigram_feat = pos_bigram_features(proc_data, label)
    #print pos_bigram_feat
    #for i in range(len(features)):
    #    if label:
    #        features[i][0].update(pos_bigram_feat[i][0])
    #        print features[i]
    #    else:
    #        features[i].update(pos_bigram_feat[i])
    #        print features[i]
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

    classifier = nltk.classify.MaxentClassifier.train(training_set, algorithm,
            max_iter=N_ITERATIONS)


    dev_set = get_dev_set()
    labels = classifier.batch_classify(dev_set[0])
    print labels

    correct = 0
    true_male = 0
    true_female = 0
    false_male = 0
    false_female = 0
    for i in range(len(labels)):
        if labels[i] == dev_set[1][i]:
            correct += 1
            if labels[i] == 'M':
                true_male += 1
            else:
                true_female += 1
        else:
            if labels[i] == 'M':
                false_female += 1
            else:
                false_male += 1
    print('Accuracy: ' + str(float(correct)/len(labels)))
    print('Precision: ' + str(float(true_male) / (true_male + false_male)))
    print('Recall: ' + str(float(true_male) / (true_male + false_female)))

if __name__ == '__main__':
    classify()
