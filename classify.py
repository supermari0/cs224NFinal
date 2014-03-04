import collections
import math
import nltk
from optparse import OptionParser
import pickle
import sklearn
from sklearn.svm import LinearSVC
import string

TRAIN_DATA_FILE = 'data/proc_train'
DEV_DATA_FILE = 'data/proc_dev'
TEST_DATA_FILE = 'data/proc_test'

COMMON_BIGRAMS_FILE = 'data/top_bigrams'
COMMON_TOKENS_FILE = 'data/top_tokens'
COMMON_TRIGRAMS_FILE = 'data/top_trigrams'

POS_FILE = 'data/pos_set'
POS_BIGRAMS_FILE = 'data/pos_bigrams'
POS_TRIGRAMS_FILE = 'data/pos_trigrams'

MALE_NAMES_FILE = 'data/male_names.txt'
FEMALE_NAMES_FILE = 'data/female_names.txt'

N_ITERATIONS = 50

def len_features(proc_data, label):
    """ This feature puts a speech into 1 of 6 buckets depending on the speech
    length. Discretization of this feature appears to work better than simply
    using the raw length as a feature. """
    features = []

    for speech_tuple in proc_data:
        token_tuples = speech_tuple[0]

        speech_len = len(token_tuples)

        if speech_len < 500:
            speech_len = 1
        elif speech_len < 1000:
            speech_len = 2
        elif speech_len < 2000:
            speech_len = 3
        elif speech_len < 3000:
            speech_len = 4
        elif speech_len < 4000:
            speech_len = 5
        else:
            speech_len = 6

        if label:
            gender_tag = speech_tuple[1]
            features.append(({'speech_len' : speech_len}, gender_tag))
        else:
            features.append({'speech_len' : speech_len})

    return features

def pos_bigram_features(proc_data, label):
    """ Binary features for part-of-speech bigrams. Only finds bigrams that
    were precomputed from the training data (data/proc_train). This means that
    if the training data changes, the bigrams looked for will be the same. """
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
    """ Helper function for pos_bigram_features. """
    for i in range(1, len(token_tuples)):
        pos = token_tuples[i][1]
        prev_pos = token_tuples[i-1][1]
        pos_bigram = prev_pos + '+' + pos
        if pos_bigram in pos_bigrams_dict:
            pos_bigrams_dict[pos_bigram] = 1
    return pos_bigrams_dict

def pos_trigram_features(proc_data, label):
    """ Binary features for part-of-speech trigrams. Only finds trigrams that
    were precomputed from the training data (data/proc_train). This means that
    if the training data changes, the trigrams looked for will be the same. """
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
    """ Helper function for pos_trigram_features. """
    for i in range(2, len(token_tuples)):
        pos = token_tuples[i][1]
        prev_pos = token_tuples[i-1][1]
        prev2_pos = token_tuples[i-2][1]
        pos_trigram = prev2_pos + '+' + prev_pos + '+' + pos
        if pos_trigram in pos_trigrams_dict:
            pos_trigrams_dict[pos_trigram] = 1
    return pos_trigrams_dict

def common_bigram_features(proc_data, label):
    """ Binary features for bigrams. The bigrams searched for consist of the
    1500 most common bigrams in data/proc_train. """
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
    """ Helper function for common_bigram_features. """
    for i in range(1, len(token_tuples)):
        token = token_tuples[i][0]
        prev_token = token_tuples[i-1][0]
        bigram = prev_token + '+' + token
        if bigram in common_bigrams_dict:
            common_bigrams_dict[bigram] = 1

    return common_bigrams_dict

def common_trigram_features(proc_data, label):
    """ Binary features for trigrams. The trigrams searched for consist of the
    750 most common trigrams in data/proc_train. """
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
    """ Helper function for common_trigram_features. """
    for i in range(2, len(token_tuples)):
        token = token_tuples[i][0]
        prev_token = token_tuples[i-1][0]
        prev2_token = token_tuples[i-2][0]
        trigram = prev2_token + '+' + prev_token + '+' + token
        if trigram in common_trigrams_dict:
            common_trigrams_dict[trigram] = 1

    return common_trigrams_dict

def male_name_features(proc_data, label):
    """ Binary features for whether or not a speech contains a given male
    Shakespeare name. The names were taken from a Shakespeare name list at
    http://www.namenerds.com/uucn/shakes.html. """
    features = []

    male_names_set = set()
    male_name_file = open(MALE_NAMES_FILE, 'r')

    for name in male_name_file:
        male_names_set.add(name.lower().strip())

    for speech_tuple in proc_data:
        token_tuples = speech_tuple[0]

        male_name_count = 0

        for token_pos in token_tuples:
            token = token_pos[0]
            
            if token in male_names_set:
                male_name_count += 1

        if label:
            gender_tag = speech_tuple[1]
            features.append(({'n_male_names' : male_name_count}, gender_tag))
        else:
            features.append({'n_male_names' : male_name_count})

    return features

def female_name_features(proc_data, label):
    """ Binary features for whether or not a speech contains a given female
    Shakespeare name. The names were taken from a Shakespeare name list at
    http://www.namenerds.com/uucn/shakes.html. """
    features = []

    female_names_set = set()
    female_name_file = open(FEMALE_NAMES_FILE, 'r')

    for name in female_name_file:
        female_names_set.add(name.lower().strip())

    for speech_tuple in proc_data:
        token_tuples = speech_tuple[0]

        female_name_count = 0

        for token_pos in token_tuples:
            token = token_pos[0]
            
            if token in female_names_set:
                female_name_count += 1

        if label:
            gender_tag = speech_tuple[1]
            features.append(({'n_female_names' : female_name_count}, gender_tag))
        else:
            features.append({'n_female_names' : female_name_count})

    return features

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

def mine_features(proc_data, label):
    """ A feature for the number of times the speaker uses the word 'mine'. The
    idea behind this feature was that men and women might differ in their use
    of possessives in Shakespeare. """
    features = []

    for speech_tuple in proc_data:
        token_tuples = speech_tuple[0]

        mine_count = 0
        for (token, pos) in token_tuples:
            if token == 'mine':
                mine_count += 1

        mine_count_log = math.log(float(mine_count + 1) / (len(token_tuples) +
            len(proc_data)))

        if label:
            gender_tag = speech_tuple[1]
            features.append(({'n_mine': mine_count_log}, gender_tag))
        else:
            features.append({'n_mine': mine_count_log})

    return features

def thine_features(proc_data, label):
    """ A feature for the number of times the speaker uses the word 'thine'. The
    idea behind this feature was that men and women might differ in their use
    of possessives in Shakespeare. """
    features = []

    for speech_tuple in proc_data:
        token_tuples = speech_tuple[0]

        thine_count = 0
        for (token, pos) in token_tuples:
            if token == 'thine':
                thine_count += 1

        thine_count_log = math.log(float(thine_count + 1) / (len(token_tuples) +
            len(proc_data)))

        if label:
            gender_tag = speech_tuple[1]
            features.append(({'n_thine': thine_count}, gender_tag))
        else:
            features.append({'n_thine': thine_count})

    return features


def extract_features(proc_data, label, algorithm):
    """ Returns an array of feature dictionaries from proc_data. If the data is
    labeled as indicated by the label boolean, the data will be labeled.
    algorithm adjusts the features used based on the selected algorithm (SVM
    works better with more features). """
    features = common_bigram_features(proc_data, label)
    pos_trigram_feat = pos_trigram_features(proc_data, label)
    len_feat = len_features(proc_data, label)

    if algorithm == 'SVM':
        trigram_feat = common_trigram_features(proc_data, label)
        male_name_feat = male_name_features(proc_data, label)
        female_name_feat = female_name_features(proc_data, label)
        token_feat = common_token_features(proc_data, label)
        mine_feat = mine_features(proc_data, label)
        thine_feat = thine_features(proc_data, label)

    for i in range(len(features)):
        if label:
            features[i][0].update(pos_trigram_feat[i][0])
            features[i][0].update(len_feat[i][0])
            if algorithm == 'SVM':
                features[i][0].update(trigram_feat[i][0])
                features[i][0].update(male_name_feat[i][0])
                features[i][0].update(female_name_feat[i][0])
                features[i][0].update(token_feat[i][0])
                features[i][0].update(mine_feat[i][0])
                features[i][0].update(thine_feat[i][0])
                #features[i] = (combine_svm_bin_features(features[i][0]),
                #        features[i][1])
        else:
            features[i].update(pos_trigram_feat[i])
            features[i].update(len_feat[i])
            if algorithm == 'SVM':
                features[i].update(trigram_feat[i])
                features[i].update(male_name_feat[i])
                features[i].update(female_name_feat[i])
                features[i].update(token_feat[i])
                features[i].update(mine_feat[i])
                features[i].update(thine_feat[i])
                #features[i] = combine_svm_bin_features(features[i])
    return features

def binary_feature(feature_val_pair):
    """ Boolean function returning true iff the feature is a binary feature
    (one which only accepts values of 0 or 1). """
    feat_name = feature_val_pair[0]

    return ((feat_name != 'speech_len') and (feat_name != 'n_male_names') and
            (feat_name != 'n_female_names') and (feat_name != 'n_mine') and
            (feat_name != 'n_thine'))


def combine_svm_bin_features(features):
    """ Combines binary features quadratically for use in a SVM. For example,
    if a speech contains the bigram "the hat" and the trigram "the hat is", a
    new feature representing both of those features will be set to 1. If the
    speech contains "the hat" but not "the hat is", then the new feature will
    be set to 0. """
    feature_list = features.items()

    new_features = []
    for i in range(len(feature_list)):
        original_feature = feature_list[i]
        if binary_feature(original_feature):
            for j in range(i+1, len(feature_list)):
                feature_to_combine = feature_list[j]
                if binary_feature(feature_to_combine):
                    if original_feature[1] == 1 and feature_to_combine[1] == 1:
                        new_feat = (original_feature[0] + '/' +
                                feature_to_combine[0], 1)
                        new_features.append(new_feat)

    feature_list = feature_list + new_features
    return dict(feature_list)

def extract_labels(proc_data):
    """ Returns an array of labels from the labeled data. 'M' and 'F' are the
    only valid labels. """
    labels = []
    for speech_tuple in proc_data:
        labels += speech_tuple[1]
    return labels

def get_all_data():
    """ Returns all the processed data concatenated together. """
    train_pickled = open(TRAIN_DATA_FILE, 'rb')
    train_proc_data = pickle.load(train_pickled)
    train_pickled.close()

    dev_pickled = open(DEV_DATA_FILE, 'rb')
    dev_proc_data = pickle.load(dev_pickled)
    dev_pickled.close()

    test_pickled = open(TEST_DATA_FILE, 'rb')
    test_proc_data = pickle.load(test_pickled)
    test_pickled.close()

    return train_proc_data + dev_proc_data + test_proc_data

def get_training_set(algorithm):
    """ Extracts features and returns the training set ready to be fed into a
    given algorithm. """
    train_pickled = open(TRAIN_DATA_FILE, 'rb')
    train_proc_data = pickle.load(train_pickled)
    train_pickled.close()

    featureset = extract_features(train_proc_data, True,
        algorithm)

    return featureset

def get_dev_set(algorithm):
    """ Returns a tuple whose first element is the unlabeled list of feature
    dicts ready for classification and whose second element is the list of
    gold labels. """
    dev_pickled = open(TEST_DATA_FILE, 'rb')
    dev_proc_data = pickle.load(dev_pickled)
    dev_pickled.close()

    featureset = extract_features(dev_proc_data, False, algorithm)
    labels = extract_labels(dev_proc_data)

    return (featureset, labels)

def get_test_set(algorithm):
    """ Returns a tuple whose first element is the unlabeled list of feature
    dicts ready for classification and whose second element is the list of
    gold labels. """
    test_pickled = open(TEST_DATA_FILE, 'rb')
    test_proc_data = pickle.load(dev_pickled)
    test_pickled.close()

    featureset = extract_features(test_proc_data, False, algorithm)
    labels = extract_labels(test_proc_data)

    return (featureset, labels)

def get_k_datasets(proc_data, k, algorithm, kfold):
    """ Returns the kth training and test sets for k-fold cross-validation. k
    is 0-indexed. Training set is first member of list, test set is
    second member. Test set is a tuple whose first member is the unlabeled data
    ready for classification and whose second member is the list of gold
    labels. """
    start_index = k * (len(proc_data) / kfold)

    if k == kfold - 1:
        train_data = proc_data[start_index:]
        test_data = proc_data[0:start_index]
    else:
        end_index = (k + 1) * (len(proc_data) / kfold)
        train_data = proc_data[start_index:end_index]
        test_data = proc_data[0:start_index] + proc_data[end_index:]

    train_featureset = extract_features(train_data, True, algorithm)
    test_featureset = extract_features(test_data, False, algorithm)
    test_labels = extract_labels(test_data)
     
    return [train_featureset, (test_featureset, test_labels)]

def results_from_labels(hypothesis, gold):
    """ Calculates results from classifier hypothesis and gold labels. """

    correct = 0
    true_male = 0
    true_female = 0
    false_male = 0
    false_female = 0

    for i in range(len(hypothesis)):
        if hypothesis[i] == gold[i]:
            correct += 1
            if gold[i] == 'M':
                true_male += 1
            else:
                true_female += 1
        else:
            if gold[i] == 'M':
                false_female += 1
            else:
                false_male += 1

    accuracy = float(correct)/len(gold)
    precision = float(true_male) / (true_male + false_male)
    recall = float(true_male) / (true_male + false_female)
    f1_score = 2 * ( (precision * recall) / (precision + recall))

    return [accuracy, precision, recall, f1_score]


def classify_results(training_set, test_set, test_labels, algo_choice):
    """ Trains the chosen algorithm and returns the array of results. """
    if algo_choice is None or algo_choice == 'MaxEnt':
        print('Training classifier with MaxEnt algorithm...')
        # Set algorithm to GIS because of bug in scipy (missing maxentropy module).
        algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]

        classifier = nltk.classify.MaxentClassifier.train(training_set, algorithm,
                max_iter=N_ITERATIONS)
    elif algo_choice == 'NaiveBayes':
        print('Training classifier with NaiveBayes algorithm...')
        classifier = nltk.classify.NaiveBayesClassifier.train(training_set)
    elif algo_choice == 'SVM':
        print('Training classifier with SVM algorithm...')
        classifier = nltk.classify.SklearnClassifier(LinearSVC())
        classifier.train(training_set)

    classify_labels = classifier.batch_classify(test_set)

    return results_from_labels(classify_labels, test_labels)

def print_results(results, test_mode):
    """ Prints the result stats from classification. If we are performing
    k-fold cross-validation, we average the statistics over the k runs. """
    if test_mode == 'kfold':
        accuracy = float(sum(result[0] for result in results)) / len(results)
        precision = float(sum(result[1] for result in results)) / len(results)
        recall = float(sum(result[2] for result in results)) / len(results)
        f1_score = float(sum(result[3] for result in results)) / len(results)
    else:
        accuracy = results[0]
        precision = results[1]
        recall = results[2]
        f1_score = results[3]

    print('Accuracy: ' + str(accuracy))
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1 Score: ' + str(f1_score))


def classify():
    """ Main method. """

    parser = OptionParser()
    parser.add_option('-a', '--algorithm', dest='algorithm', help='sets' +
        ' algorithm to one of NaiveBayes, SVM, or MaxEnt. defaults to MaxEnt')
    parser.add_option('-t', '--test', dest='test', help='sets test mode '
            + 'to one of kfold, dev, or test. kfold parameter must be'
            + ' specified if kfold mode is selected. defaults to test')
    parser.add_option('-k', '--kfold', dest='kfold', help='sets number of '
            + 'folds for k-fold cross validation. defaults to 5 if kfold '
            + 'selected.')
            
    (options, args) = parser.parse_args()

    algo_choice = options.algorithm

    if options.test is None or options.test == 'test':
        print('Classifying test set...')
        training_set = get_training_set(algo_choice)
        test_data = get_test_set(algo_choice)

        test_set = test_data[0]
        test_labels = test_data[1]

        results = classify_results(training_set, test_set, test_labels,
                algo_choice)

    elif options.test == 'dev':
        print('Classifying dev set...')
        training_set = get_training_set(algo_choice)
        test_data = get_dev_set(algo_choice)

        test_set = test_data[0]
        test_labels = test_data[1]

        results = classify_results(training_set, test_set, test_labels,
                algo_choice)

    elif options.test == 'kfold':
        if options.kfold is None:
            kfold = 5
        else:
            try:
                kfold = int(options.kfold)
            except ValueError:
                print('Invalid integer for kfold. Defaulting to 5 folds.')
                kfold = 5

        print('Classifying entire dataset with ' + str(kfold) + '-fold cross-validation...')
        proc_data = get_all_data()

        k_results = []

        for i in range(kfold):
            print('Calculating ' + str(i) + ' fold...')
            datasets = get_k_datasets(proc_data, i, algo_choice, kfold)
            training_set = datasets[0]
            test_set = datasets[1][0]
            test_labels = datasets[1][1]
            results = classify_results(training_set, test_set, test_labels,
                    algo_choice)
            k_results.append(results)

        results = k_results

    print_results(results, options.test)

if __name__ == '__main__':
    classify()
