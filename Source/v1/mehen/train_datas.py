words = [
    "how","are","you","hi","hello","ai","i","am","they","we",
    "what","is","your","name","my","good","bad","fine","thanks",
    "who","and","or","milk","coffee","tea","with","esc","where",
    "say","says","tell","tells","said","have","has","he","she","it",
    "cold","hot","when",""
]
numbers = list(range(0,len(words)))
word2id = {a:b for a,b in zip(words,numbers)}

if __name__ == "__main__":
    print(word2id)
    
id2word = {v: k for k, v in word2id.items()}
vocab_size = len(word2id)

def encode(text):
    return [word2id[w] for w in text.split() if w in word2id]

# TRAINING DATA
data = [
    # how are you
    ([0], 1),
    ([0,1], 2),
    ([0,1,2], 26),
    # hi -> how are you
    ([3], 0),
    ([3,0], 1),
    ([3,0,1], 2),
    ([3,0,1,2], 26),

    # hello -> hi
    ([4], 3),
    ([4,3], 26),

    # i am ai
    ([6], 7),
    ([6,7], 5),
    ([6,7,5], 26),

    # they are good
    ([8], 1),
    ([8,1], 15),
    ([8,1,15], 26),

    # we are fine
    ([9], 1),
    ([9,1], 17),
    ([9,1,17], 26),

    # what is your name
    ([10], 11),
    ([10,11], 12),
    ([10,11,12], 13),
    ([10,11,12,13], 26),

    # my name ai
    ([14], 13),
    ([14,13], 5),
    ([14,13,5], 26),

    # how are you -> fine
    ([0,1,2], 17),
    ([0,1,2,17], 26),

    # how are you -> good
    ([0,1,2], 15),
    ([0,1,2,15], 26),

    # how are you -> bad
    ([0,1,2], 16),
    ([0,1,2,16], 26),

    # thanks
    ([18], 26),

    # who are you
    ([19], 1),
    ([19,1], 2),
    ([19,1,2], 26),
    #coffee or tea
    ([23],21),
    ([23,21],24),
    ([23,21,24],26),
    #tea with milk
    ([24],25),
    ([24,25],22),
    ([24,25,22],26),
    # i have said
    ([6,33],32),
    ([6,33,32],26),
    #coffee is hot
    ([23,11],39),
    ([23,11,39],26)
]




