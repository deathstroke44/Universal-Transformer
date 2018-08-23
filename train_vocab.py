import pickle
from data.vocab.word import WordVocab

for task in range(1, 21):
    path = "babi-qa/task%d_train.txt" % task
    vocab_path = "babi-qa/vocab/task%d_vocab.pkl" % task
    texts = [" ".join(line[line.find(" ") + 1:-1].split("\t")[:-1]) for line in open(path)]
    print(texts)
    word_vocab = WordVocab(texts)
    with open(vocab_path, "wb") as f:
        pickle.dump(word_vocab, f)
