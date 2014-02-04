from collections import defaultdict
import operator
import pickle

if __name__ == '__main__':
    word_counts = defaultdict(lambda: 0)
    male_word_counts = defaultdict(lambda: 0)
    female_word_counts = defaultdict(lambda: 0)

    pos_counts = defaultdict(lambda: 0)

    male_bigram_counts = defaultdict(lambda: 0)
    female_bigram_counts = defaultdict(lambda: 0)

    male_trigram_counts = defaultdict(lambda: 0)
    female_trigram_counts = defaultdict(lambda: 0)

    proc_train_pickled = open('data/proc_train', 'rb')
    proc_train = pickle.load(proc_train_pickled)
    proc_train_pickled.close()

    n_tokens = 0
    n_male_tokens = 0
    n_female_tokens = 0

    for speech_tuple in proc_train:
        token_tuples = speech_tuple[0]
        gender_tag = speech_tuple[1]
        for i in range(len(token_tuples)):
            token_tuple = token_tuples[i]
            token = token_tuple[0]

            pos = token_tuple[1]
            word_counts[token] += 1
            pos_counts[pos] += 1
            n_tokens += 1

            if i > 0:
                prev_token = token_tuples[i-1][0]
                bigram = prev_token + '+' + token

            if i > 1:
                prev2_token = token_tuples[i-2][0]
                trigram = prev2_token + '+' + prev_token + '+' + token

            if gender_tag == 'M':
                male_word_counts[token] += 1
                n_male_tokens += 1
                
                if i > 0:
                    male_bigram_counts[bigram] += 1    
                if i > 1:
                    male_trigram_counts[trigram] += 1
            else:
                female_word_counts[token] += 1
                n_female_tokens += 1

                if i > 0:
                    female_bigram_counts[bigram] += 1    
                if i > 1:
                    female_trigram_counts[trigram] += 1


    #sorted_male = sorted(male_word_counts.iteritems(),
    #    key=operator.itemgetter(1), reverse=True)

    #print('Top 2000 sorted male tokens: ')
    #print(sorted_male[:2000])

    #print('\n')

    #sorted_female = sorted(female_word_counts.iteritems(), key=operator.itemgetter(1),
    #    reverse=True)

    #print('Top 2000 sorted female tokens: ')
    #print(sorted_female[:2000])

    #print('\n')

    #print('Number of tokens: ' + str(n_tokens))
    #print('Number of male tokens: ' + str(n_male_tokens))
    #print('Number of female tokens: ' + str(n_female_tokens))

    sorted_male_bigrams = sorted(male_bigram_counts.iteritems(),
            key=operator.itemgetter(1), reverse=True)
    sorted_female_bigrams = sorted(female_bigram_counts.iteritems(),
            key=operator.itemgetter(1), reverse=True)

    print('Top 500 sorted male bigrams: ')
    print(sorted_male_bigrams[:500])

    print('Top 500 sorted female bigrams: ')
    print(sorted_female_bigrams[:500])

    print('Number of bigrams when merged: ' + str(len(set(sorted_male_bigrams[:500]
        + sorted_female_bigrams[:500]))))
 
    top_bigrams = set([bigram for (bigram, count) in sorted_male_bigrams[:500]] +
            [bigram for (bigram, count) in sorted_female_bigrams[:500]])

    top_bigram_file = open('data/top_bigrams', 'wb')
    pickle.dump(top_bigrams, top_bigram_file)
    top_bigram_file.close()

    #top_tokens = set([token for (token, count) in sorted_male[:2000]] + [token for
    #    (token, count) in sorted_female[:2000]])

    #print('Number of top tokens (intersection of top 2K M/F): ' +
    #        str(len(top_tokens)))

    #top_token_file = open('data/top_tokens', 'wb')
    #pickle.dump(top_tokens, top_token_file)
    #top_token_file.close()

    #parts_of_speech = set([pos for (pos, count) in pos_counts.items()])

    #pos_file = open('data/pos_set', 'wb')
    #pickle.dump(parts_of_speech, pos_file)
    #pos_file.close()
