import nltk

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
    train_speeches = eval(open('data/train', 'r').read())
    dev_speeches = eval(open('data/dev', 'r').read())
    test_speeches = eval(open('data/test', 'r').read())
    
    proc_train_speeches = process_speeches(train_speeches)
    proc_dev_speeches = process_speeches(dev_speeches)
    proc_test_speeches = process_speeches(test_speeches)

    proc_train = open('data/proc_train' 'w')
    proc_dev = open('data/proc_dev', 'w')
    proc_test = open('data/proc_test', 'w')

    proc_train.write(str(proc_train_speeches))
    proc_train.write(str(proc_dev_speeches))
    proc_train.write(str(proc_test_speeches))
