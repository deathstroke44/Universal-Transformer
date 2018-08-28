from data.dataset.qa_task import BabiQADataset
from data.vocab.word import WordVocab
from torch.utils.data import DataLoader

from model.qa_transformer import UniversalTransformer
from trainer.qa_universal import UniversalTransformerQATrainer

import torch
import nsml

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
word_vocab = WordVocab.load_vocab("babi-qa/vocab/task1_vocab.pkl")
dataset = {
    "train": BabiQADataset("babi-qa/task1_train.txt", word_vocab, word_vocab, story_len=14, seq_len=6),
    "test": BabiQADataset("babi-qa/task1_test.txt", word_vocab, word_vocab, story_len=14, seq_len=6)
}
dataloader = {
    "train": DataLoader(dataset["train"], batch_size=64, shuffle=True),
    "test": DataLoader(dataset["test"], batch_size=64)
}

model = UniversalTransformer(enc_seq_len=14, dec_seq_len=1, d_model=128, n_enc_vocab=len(word_vocab),
                             n_dec_vocab=len(word_vocab), h=4, t_steps=4, dropout=0.2).to(device)

epochs = 100

trainer = UniversalTransformerQATrainer(model, dataloader, device)

for epoch in range(epochs):
    train_loss, train_acc = trainer.train(epoch)
    test_loss, test_acc = trainer.test(epoch)
    nsml.report(step=epoch, ep_train_loss=train_loss, ep_train_acc=train_acc,
                ep_test_loss=test_loss, ep_test_acc=test_acc)
