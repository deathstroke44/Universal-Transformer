from data.dataset.qa_task import BabiQADataset
from data.vocab.word import WordVocab
from torch.utils.data import DataLoader

from model.qa_transformer import UniversalTransformer
from trainer.qa_transformer import UniversalTransformerQATrainer

from params import *
import argparse
import torch
import nsml
parser = argparse.ArgumentParser(description='Train QA.')

parser.add_argument('-t','--task',default='task1')

# from params import batch_size, model_dim, h, t_steps, dropout
args = parser.parse_args()
task=args.task
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
word_vocab = WordVocab.load_vocab("babi-qa/vocab/"+task+"_vocab.pkl")
answer_vocab = WordVocab.load_vocab("babi-qa/vocab/"+task+"_answer_vocab.pkl")

dataset = {
    "train": BabiQADataset("babi-qa/"+task+"_train.txt", word_vocab, answer_vocab, story_len=14, seq_len=6),
    "test": BabiQADataset("babi-qa/"+task+"_test.txt", word_vocab, answer_vocab, story_len=14, seq_len=6)
}

model = UniversalTransformer(enc_seq_len=14, dec_seq_len=1, d_model=model_dim, n_enc_vocab=len(word_vocab),
                             n_dec_vocab=len(word_vocab), h=h, t_steps=t_steps, dropout=dropout).to(device)

dataloader = {
    "train": DataLoader(dataset["train"], batch_size=batch_size, shuffle=True),
    "test": DataLoader(dataset["test"], batch_size=batch_size)
}

trainer = UniversalTransformerQATrainer(model, dataloader, device)

epochs = 300
for epoch in range(epochs):
    train_loss, train_acc = trainer.train(epoch)
    test_loss, test_acc = trainer.test(epoch)
    nsml.report(step=epoch, ep_train_loss=train_loss, ep_train_acc=train_acc,
                ep_test_loss=test_loss, ep_test_acc=test_acc)
    print("[TEST RESULT] Epoch %d, ACC: %.2f Loss: %.4f" % (epoch, test_acc, test_loss))
