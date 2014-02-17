from collections import defaultdict
import operator
import pickle

if __name__ == '__main__':
    word_counts = defaultdict(lambda: 0)
    male_word_counts = defaultdict(lambda: 0)
    female_word_counts = defaultdict(lambda: 0)

    pos_counts = defaultdict(lambda: 0)

    m_pos_bigram_counts = defaultdict(lambda: 0)
    m_pos_trigram_counts = defaultdict(lambda: 0)

    f_pos_bigram_counts = defaultdict(lambda: 0)
    f_pos_trigram_counts = defaultdict(lambda: 0)

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

                prev_pos = token_tuples[i-1][1]
                pos_bigram = prev_pos + '+' + pos

            if i > 1:
                prev2_token = token_tuples[i-2][0]
                trigram = prev2_token + '+' + prev_token + '+' + token

                prev2_pos = token_tuples[i-1][1]
                pos_trigram = prev2_pos + '+' + prev_pos + '+' + pos

            if gender_tag == 'M':
                male_word_counts[token] += 1
                n_male_tokens += 1
                
                if i > 0:
                    male_bigram_counts[bigram] += 1
                    m_pos_bigram_counts[pos_bigram] += 1
                if i > 1:
                    male_trigram_counts[trigram] += 1
                    m_pos_trigram_counts[pos_trigram] += 1
            else:
                female_word_counts[token] += 1
                n_female_tokens += 1

                if i > 0:
                    female_bigram_counts[bigram] += 1
                    f_pos_bigram_counts[pos_bigram] += 1
                if i > 1:
                    female_trigram_counts[trigram] += 1
                    f_pos_trigram_counts[pos_trigram] += 1


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

    #sorted_male_bigrams = sorted(male_bigram_counts.iteritems(),
    #        key=operator.itemgetter(1), reverse=True)
    #sorted_female_bigrams = sorted(female_bigram_counts.iteritems(),
    #        key=operator.itemgetter(1), reverse=True)

    #print('Top 750 sorted male bigrams: ')
    #print(sorted_male_bigrams[:750])

    #print('Top 750 sorted female bigrams: ')
    #print(sorted_female_bigrams[:750])

    #print('Number of bigrams when merged: ' + str(len(set(sorted_male_bigrams[:750]
    #    + sorted_female_bigrams[:750]))))
 
    #top_bigrams = set([bigram for (bigram, count) in sorted_male_bigrams[:750]] +
    #        [bigram for (bigram, count) in sorted_female_bigrams[:750]])

    #top_bigram_file = open('data/top_bigrams', 'wb')
    #pickle.dump(top_bigrams, top_bigram_file)
    #top_bigram_file.close()

    #sorted_male_trigrams = sorted(male_trigram_counts.iteritems(),
    #        key=operator.itemgetter(1), reverse=True)
    #sorted_female_trigrams = sorted(female_trigram_counts.iteritems(),
    #        key=operator.itemgetter(1), reverse=True)

    #print('Top 750 sorted male trigrams: ')
    #print(sorted_male_trigrams[:750])

    #print('Top 750 sorted female trigrams: ')
    #print(sorted_female_trigrams[:750])

    #print('Number of trigrams when merged: ' + str(len(set(sorted_male_trigrams[:750]
    #    + sorted_female_trigrams[:750]))))
 
    #top_trigrams = set([trigram for (trigram, count) in sorted_male_trigrams[:750]] +
    #        [trigram for (trigram, count) in sorted_female_trigrams[:750]])

    #top_trigram_file = open('data/top_trigrams', 'wb')
    #pickle.dump(top_trigrams, top_trigram_file)
    #top_trigram_file.close()

    #sorted_m_pos_bigrams = sorted(m_pos_bigram_counts.iteritems(),
    #        key=operator.itemgetter(1), reverse=True)
    #sorted_f_pos_bigrams = sorted(f_pos_bigram_counts.iteritems(),
    #        key=operator.itemgetter(1), reverse=True)

    #print('M POS bigrams: ' + str(sorted_m_pos_bigrams))
    #print('F POS bigrams: ' + str(sorted_f_pos_bigrams))

    #pos_bigrams = set([bigram for (bigram, count) in sorted_m_pos_bigrams]
    #        + [bigram for (bigram, count) in sorted_f_pos_bigrams])

    #pos_bigram_file = open('data/pos_bigrams', 'wb')
    #pickle.dump(pos_bigrams, pos_bigram_file)
    #pos_bigram_file.close()

    sorted_m_pos_trigrams = sorted(m_pos_trigram_counts.iteritems(),
            key=operator.itemgetter(1), reverse=True)
    sorted_f_pos_trigrams = sorted(f_pos_trigram_counts.iteritems(),
            key=operator.itemgetter(1), reverse=True)

    print('M POS trigrams: ' + str(sorted_m_pos_trigrams))
    print('F POS trigrams: ' + str(sorted_f_pos_trigrams))

    pos_trigrams = set([trigram for (trigram, count) in sorted_m_pos_trigrams]
            + [trigram for (trigram, count) in sorted_f_pos_trigrams])

    pos_trigram_file = open('data/pos_trigrams', 'wb')
    pickle.dump(pos_trigrams, pos_trigram_file)
    pos_trigram_file.close()

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
