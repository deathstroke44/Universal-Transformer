import torch
import nsml

from torch.utils.data import DataLoader
from data.dataset.dialog.task1 import Task1DataSet
from data.vocab import WordVocab

from params import *
from model.universal_transformer import UniversalTransformer
from trainer.dialog_transformer import UniversalTransformerQATrainer

enc_vocab = WordVocab.load_vocab("dataset/babi-dialog/vocab/task1_word_vocab.pkl")
dec_vocab = WordVocab.load_vocab("dataset/babi-dialog/vocab/task1_word_vocab.pkl")
kb_vocab = WordVocab.load_vocab("dataset/babi-dialog/vocab/kb_word_vocab.pkl")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

enc_vocab_size = len(enc_vocab)
dec_vocab_size = len(dec_vocab)
print("ECD_VOCAB: %d, DEC_VOCAB: %d" % (enc_vocab_size, dec_vocab_size))

history_len = 16
seq_len = 10

dataset = {
    "train": Task1DataSet("dataset/babi-dialog/task1_train.txt", enc_vocab, dec_vocab,
                          seq_len, seq_len, history_len=history_len),
    "test": Task1DataSet("dataset/babi-dialog/task1_test.txt", enc_vocab, dec_vocab,
                         seq_len, seq_len, history_len=history_len)
}

dataloader = {key: DataLoader(value, batch_size=batch_size) for key, value in dataset.items()}

model = UniversalTransformer(enc_seq_len=history_len, dec_seq_len=seq_len, d_model=model_dim,
                             n_enc_vocab=enc_vocab_size,
                             n_dec_vocab=dec_vocab_size, h=h, t_steps=t_steps, dropout=dropout).to(device)

trainer = UniversalTransformerQATrainer(model, dataloader, device)

epochs = 300
for epoch in range(epochs):
    train_loss, train_acc = trainer.train(epoch)
    print("[TRAIN RESULT]\tEpoch %d, ACC: %.2f Loss: %.4f" % (epoch, train_acc, train_loss))
    test_loss, test_acc = trainer.test(epoch)
    print("[TEST RESULT]\tEpoch %d, ACC: %.2f Loss: %.4f" % (epoch, test_acc, test_loss))
    nsml.report(step=epoch, ep_train_loss=train_loss, ep_train_acc=train_acc,
                ep_test_loss=test_loss, ep_test_acc=test_acc)
