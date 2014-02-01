
if __name__ == '__main__':
    male_file = open('data/Male_speeches.txt', 'r')
    male_count = 0
    for line in male_file:
        if line == '\n':
            male_count += 1
    # Add 1 since EOF doesn't have newline.
    male_count += 1

    female_file = open('data/Female_speeches.txt', 'r')
    female_count = 0
    for line in female_file:
        if line == '\n':
            female_count += 1
    # Add 1 since EOF doesn't have newline.
    female_count += 1

    print(str(male_count) + ' male speeches')
    print(str(female_count) + ' female speeches')
