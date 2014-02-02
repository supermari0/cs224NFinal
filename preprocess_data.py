import nltk
import pickle

def process_speeches(speech_arr):
    proc_speeches = []

    porter = nltk.PorterStemmer()

    for (speech, tag) in speech_arr:
        tokens = nltk.word_tokenize(speech)
        pos_tagged_tokens = nltk.pos_tag(tokens)
        stemmed_tokens = [(porter.stem(t), p) for (t, p) in pos_tagged_tokens]
        lower_stemmed_tokens = [(t.lower(), p) for (t, p) in stemmed_tokens]
        proc_speeches.append((lower_stemmed_tokens, tag))

    return proc_speeches

if __name__ == '__main__':

    print('Reading raw data...')
    train_speeches = eval(open('data/train', 'r').read())
    dev_speeches = eval(open('data/dev', 'r').read())
    test_speeches = eval(open('data/test', 'r').read())
    
    print('Processing train speeches...')
    proc_train_speeches = process_speeches(train_speeches)
    print('Processing dev speeches...')
    proc_dev_speeches = process_speeches(dev_speeches)
    print('Processing test speeches...')
    proc_test_speeches = process_speeches(test_speeches)

    print('Writing processed data to disk...')
    proc_train = open('data/proc_train', 'wb')
    proc_dev = open('data/proc_dev', 'wb')
    proc_test = open('data/proc_test', 'wb')

    pickle.dump(proc_train_speeches, proc_train)
    pickle.dump(proc_dev_speeches, proc_dev)
    pickle.dump(proc_test_speeches, proc_test)

    print('Processing done!')
