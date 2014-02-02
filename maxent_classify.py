import nltk
import nltk.classify.maxent as maxent
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
    key token in the text. """

    for (token, pos) in token_tuples:
        if token in common_tokens_dict:
            common_tokens_dict[token] += 1

    n_tokens = len(token_tuples)
    for token, count in common_tokens_dict.items():
        common_tokens_dict[token] = count / float(n_tokens)

    return common_tokens_dict

def common_token_features_labeled(proc_data):
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
        gender_tag = speech_tuple[1]

        features.append((common_token_feature_dict(common_tokens_dict, token_tuples),
            gender_tag))

    print(len(features))
    return features

def extract_labeled_features(proc_data):
    features = []
    features += common_token_features_labeled(proc_data)    
    return features


def get_training_set():
    train_pickled = open(TRAIN_DATA_FILE, 'rb')
    train_proc_data = pickle.load(train_pickled)
    train_pickled.close()
   
    featureset = extract_labeled_features(train_proc_data)

    return featureset

def classify():
    training_set = get_training_set()
    #TODO fix below once features are extracted
    #MaxEntClassifier = maxent.MaxentClassifier.train(...)
    
if __name__ == '__main__':
    classify()
