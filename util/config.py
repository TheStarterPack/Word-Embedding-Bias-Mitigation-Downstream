LM_SEQ_LEN = 50
LM_SEQ_MIN_LEN = 10

CONLL2003_SEN_LEN = 39
CONLL2003_MIN_SEN_LEN = 7

PADDING_WORD = "<pad>"
NER_TAGS = {0: '0: O', 1: '1: B-PER', 2: '2: I-PER', 3: '3: B-ORG', 4: '4: I-ORG', 5: '5: B-LOC', 6: '6: I-LOC',
            7: '7: B-MISC', 8: '8: I-MISC', 9: '9: PAD'}
POS_TAGS = {0: '"', 1: "''", 2: '#', 3: '$', 4: '(', 5: ')', 6: ',', 7: '.', 8: ':', 9: '``', 10: 'CC', 11: 'CD',
            12: 'DT', 13: 'EX', 14: 'FW', 15: 'IN', 16: 'JJ', 17: 'JJR', 18: 'JJS', 19: 'LS', 20: 'MD', 21: 'NN',
            22: 'NNP', 23: 'NNPS', 24: 'NNS', 25: 'NN|SYM', 26: 'PDT', 27: 'POS', 28: 'PRP', 29: 'PRP$', 30: 'RB',
            31: 'RBR', 32: 'RBS', 33: 'RP', 34: 'SYM', 35: 'TO', 36: 'UH', 37: 'VB', 38: 'VBD', 39: 'VBG', 40: 'VBN',
            41: 'VBP', 42: 'VBZ', 43: 'WDT', 44: 'WP', 45: 'WP$', 46: 'WRB', 47: 'PAD'}

NER_CLASSES = 10
POS_CLASSES = 48

NER_PAD_TAG_IDX = 9
POS_PAD_TAG_IDX = 47
