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

