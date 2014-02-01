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
    
    proc_train_speeches = process_speeches(train_speeches)

    print('unproc: ' + str(train_speeches[0]))
    print('proc: ' + str(proc_train_speeches[0]))
