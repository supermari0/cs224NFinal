from collections import defaultdict
import operator
import pickle

if __name__ == '__main__':
    word_counts = defaultdict(lambda: 0)
    male_word_counts = defaultdict(lambda: 0)
    female_word_counts = defaultdict(lambda: 0)


    proc_train_pickled = open('data/proc_train', 'rb')
    proc_train = pickle.load(proc_train_pickled)
    proc_train_pickled.close()

    n_tokens = 0
    n_male_tokens = 0
    n_female_tokens = 0

    for speech_tuple in proc_train:
        token_tuples = speech_tuple[0]
        gender_tag = speech_tuple[1]
        for (token, pos) in token_tuples:
            word_counts[token] += 1
            n_tokens += 1
            if gender_tag == 'M':
                male_word_counts[token] += 1
                n_male_tokens += 1
            else:
                female_word_counts[token] += 1
                n_female_tokens += 1

    sorted_male = sorted(male_word_counts.iteritems(),
        key=operator.itemgetter(1), reverse=True)

    print('Top 300 sorted male tokens: ')
    print(sorted_male[:300])

    print('\n')

    sorted_female = sorted(female_word_counts.iteritems(), key=operator.itemgetter(1),
        reverse=True)

    print('Top 500 sorted female tokens: ')
    print(sorted_female[:500])

    print('\n')

    print('Number of tokens: ' + str(n_tokens))
    print('Number of male tokens: ' + str(n_male_tokens))
    print('Number of female tokens: ' + str(n_female_tokens))

    top_tokens = set([token for (token, count) in sorted_male] + [token for
        (token, count) in sorted_female])

    top_token_file = open('data/top_tokens', 'wb')
    pickle.dump(top_tokens, top_token_file)
