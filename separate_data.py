""" This program divides the speeches into a training, development, and test
set without performing any additional processing of the data. """
import random

if __name__ == '__main__':
    male_train_remaining = 15
    male_dev_remaining = 15

    female_train_remaining = 15
    female_dev_remaining = 15

    male_speeches = []
    female_speeches = []

    male_text = open('data/Male_speeches.txt', 'r')
    
    buff = ''
    for line in male_text:
        if line == '\n':
            male_speeches.append(buff)
            buff = ''
        else:
            buff += line
    male_speeches.append(buff)
    
    male_text.close()

    female_text = open('data/Female_speeches.txt', 'r')
    buff = ''
    for line in female_text:
        if line == '\n':
            female_speeches.append(buff)
            buff = ''
        else:
            buff += line
    female_speeches.append(buff)

    female_text.close()

    print('# male speeches: ' + str(len(male_speeches)))
    print('# female speeches: ' + str(len(female_speeches)))

    male_speeches_tagged = [(speech, 'M') for speech in male_speeches]
    female_speeches_tagged = [(speech, 'F') for speech in female_speeches]

    random.shuffle(male_speeches_tagged)
    random.shuffle(female_speeches_tagged)

    train_speeches_tagged = []
    dev_speeches_tagged = []
    test_speeches_tagged = []
    while male_train_remaining > 0:
        train_speeches_tagged.append(male_speeches_tagged.pop())
        male_train_remaining -= 1
    while female_train_remaining > 0:
        train_speeches_tagged.append(female_speeches_tagged.pop())
        female_train_remaining -= 1
    while male_dev_remaining > 0:
        dev_speeches_tagged.append(male_speeches_tagged.pop())
        male_dev_remaining -= 1
    while female_dev_remaining > 0:
        dev_speeches_tagged.append(female_speeches_tagged.pop())
        female_dev_remaining -= 1

    test_speeches_tagged += male_speeches_tagged
    test_speeches_tagged += female_speeches_tagged

    random.shuffle(train_speeches_tagged)
    random.shuffle(dev_speeches_tagged)
    random.shuffle(test_speeches_tagged)

    male_train_count = 0
    female_train_count = 0
    male_dev_count = 0
    female_dev_count = 0
    male_test_count = 0
    female_test_count = 0

    for (speech, label) in train_speeches_tagged:
        if label == 'M':
            male_train_count += 1
        else:
            female_train_count += 1

    for (speech, label) in dev_speeches_tagged:
        if label == 'M':
            male_dev_count += 1
        else:
            female_dev_count += 1

    for (speech, label) in test_speeches_tagged:
       if label == 'M':
           male_test_count += 1
       else:
           female_test_count += 1
   
    print('male train: ' + str(male_train_count))
    print('female train: ' + str(female_train_count))
    print('male dev: ' + str(male_dev_count))
    print('female dev: ' + str(female_dev_count))
    print('male test: ' + str(male_test_count))
    print('female test: ' + str(female_test_count))

    train_data = open('data/train', 'w')
    train_data.write(str(train_speeches_tagged))
    train_data.close()

    dev_data = open('data/dev', 'w')
    dev_data.write(str(dev_speeches_tagged))
    dev_data.close()

    test_data = open('data/test', 'w')
    test_data.write(str(test_speeches_tagged))
    test_data.close()
