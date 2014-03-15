""" This file contains features that weren't used in the final iteration of the
project."""
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

def pos_freq_features(proc_data, label):
    pos_pickled = open(POS_FILE, 'rb')
    pos_set = pickle.load(pos_pickled)
    pos_pickled.close()

    features = []

    for speech_tuple in proc_data:
        pos_dict = dict.fromkeys(pos_set, 0)

        token_tuples = speech_tuple[0]

        for (token, pos) in token_tuples:
            if pos in pos_dict:
                pos_dict[pos] += 1

        for pos in pos_dict:
            pos_dict[pos] = (float(pos_dict[pos]) / len(token_tuples)) * 100.0

        print pos_dict
        if label:
            gender_tag = speech_tuple[1]
            features.append((pos_dict, gender_tag))
        else:
            features.append(pos_dict)
    
    return features

