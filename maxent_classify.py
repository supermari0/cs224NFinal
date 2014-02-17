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
POS_TRIGRAMS_FILE = 'data/pos_trigrams'

N_ITERATIONS = 50

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

def pos_trigram_features(proc_data, label):
    pos_trigrams_pickled = open(POS_TRIGRAMS_FILE, 'rb')
    pos_trigrams_set = pickle.load(pos_trigrams_pickled)
    pos_trigrams_pickled.close()

    features = []

    for speech_tuple in proc_data:
        pos_trigrams_dict = dict.fromkeys(pos_trigrams_set, 0)

        token_tuples = speech_tuple[0]
        pos_trigram_features = pos_trigram_feature_dict(pos_trigrams_dict,
                token_tuples)

        if label:
            gender_tag = speech_tuple[1]
            features.append((pos_trigram_features, gender_tag))
        else:
            features.append(pos_trigram_features)

    return features

def pos_trigram_feature_dict(pos_trigrams_dict, token_tuples):
    for i in range(2, len(token_tuples)):
        pos = token_tuples[i][1]
        prev_pos = token_tuples[i-1][1]
        prev2_pos = token_tuples[i-2][1]
        pos_trigram = prev2_pos + '+' + prev_pos + '+' + pos
        if pos_trigram in pos_trigrams_dict:
            pos_trigrams_dict[pos_trigram] = 1
    return pos_trigrams_dict

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



def extract_features(proc_data, label=False):
    features = common_bigram_features(proc_data, label)
    #pos_bigram_feat = pos_bigram_features(proc_data, label)
    pos_trigram_feat = pos_trigram_features(proc_data, label)
    for i in range(len(features)):
        if label:
            #features[i][0].update(pos_bigram_feat[i][0])
            features[i][0].update(pos_trigram_feat[i][0])
        else:
            #features[i].update(pos_bigram_feat[i])
            features[i].update(pos_trigram_feat[i])
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

    # Get result statistics

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

    accuracy = float(correct)/len(labels)
    precision = float(true_male) / (true_male + false_male)
    recall = float(true_male) / (true_male + false_female)
    f1_score = 2 * ( (precision * recall) / (precision + recall))

    print('Accuracy: ' + str(accuracy))
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1 Score: ' + str(f1_score))

if __name__ == '__main__':
    classify()
