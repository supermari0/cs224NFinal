import nltk

def process_speeches(speech_arr):
    proc_speeches = []

    for (speech, tag) in speech_arr:
        proc_speeches.append((nltk.word_tokenize(speech), tag))

    return proc_speeches

if __name__ == '__main__':
    train_speeches = eval(open('data/train', 'r').read())
    
    proc_train_speeches = process_speeches(train_speeches)

    print('unproc: ' + train_speeches[0][0])
    print('proc: ' + str(proc_train_speeches[0][0]))
