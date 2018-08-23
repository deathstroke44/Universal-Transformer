from data.dataset.qa_task import BabiQADataset
from data.vocab.word import WordVocab

word_vocab = WordVocab.load_vocab("babi-qa/vocab/task1_vocab.pkl")
dataset = BabiQADataset("babi-qa/task1_train.txt", word_vocab, word_vocab)

