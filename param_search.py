from data.dataset.qa_task import BabiQADataset
from data.vocab.word import WordVocab
from torch.utils.data import DataLoader

from model.qa_transformer import UniversalTransformer
from trainer.qa_transformer import UniversalTransformerQATrainer

import torch
import nsml

# from params import batch_size, model_dim, h, t_steps, dropout

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
word_vocab = WordVocab.load_vocab("babi-qa/vocab/task1_vocab.pkl")
dataset = {
    "train": BabiQADataset("babi-qa/task1_train.txt", word_vocab, word_vocab, story_len=14, seq_len=6),
    "test": BabiQADataset("babi-qa/task1_test.txt", word_vocab, word_vocab, story_len=14, seq_len=6)
}


def train_model(batch_size, model_dim, h, t, dropout, epochs=100):
    model = UniversalTransformer(enc_seq_len=14, dec_seq_len=1, d_model=model_dim, n_enc_vocab=len(word_vocab),
                                 n_dec_vocab=len(word_vocab), h=h, t_steps=t, dropout=dropout).to(device)

    dataloader = {
        "train": DataLoader(dataset["train"], batch_size=batch_size, shuffle=True),
        "test": DataLoader(dataset["test"], batch_size=batch_size)
    }

    trainer = UniversalTransformerQATrainer(model, dataloader, device)

    best_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = trainer.train(epoch)
        test_loss, test_acc = trainer.test(epoch)
        if test_acc > best_acc:
            best_acc = test_acc
        # nsml.report(step=epoch, ep_train_loss=train_loss, ep_train_acc=train_acc,
        #             ep_test_loss=test_loss, ep_test_acc=test_acc)

    return best_acc


model_dims = [16, 32, 64, 128]
dropouts = [0, 0.2, 0.4, 0.6]
batch_sizes = [16]
h_items = [2, 4, 8]
t_items = [4, 8, 16]

for batch_size in batch_sizes:
    for model_dim in model_dims:
        for h in h_items:
            for t in t_items:
                for dropout in dropouts:
                    print({"batch_size": batch_size, "model_dim": model_dim, "h": h, "t": t, "dropout": dropout})
                    best_acc = train_model(batch_size, model_dim, h, t, dropout)
                    print("Best ACC:", best_acc, "\n")
