from torch.utils.data import Dataset
from data.vocab import WordVocab
import torch


class Task1DataSet(Dataset):
    def __init__(self, dataset_path, enc_vocab, dec_vocab, q_seq_len=None, a_seq_len=None, history_len=None,
                 word_char_len=None, copy_source=None):
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab
        self.word_char_len = word_char_len
        self.copy_source = copy_source

        self.question_seq_len = q_seq_len
        self.answer_seq_len = a_seq_len
        self.history_len = history_len

        self.dialogs = self.get_data(dataset_path)

    def __getitem__(self, item):
        history, query, answer = self.dialogs[item]

        history = history + [query]
        _history = history + ["" for _ in range(self.history_len - len(history))]

        # copy_source = self.build_kb_vocab(history)
        _history = [self.enc_vocab.to_seq(his, self.question_seq_len) for his in _history][:self.history_len]
        # _query = self.enc_vocab.to_seq(query, self.question_seq_len)
        _answer = self.dec_vocab.to_seq(answer, self.answer_seq_len, with_eos=True)

        history_mask = [1 for _ in range(len(history))] + [0 for _ in range(self.history_len - len(history))]
        history, answer = torch.tensor(_history), torch.tensor(_answer)
        history_mask, answer_mask = torch.tensor(history_mask), answer.eq(0)

        return {"history": history, "answer": answer, "history_mask": history_mask, "answer_mask": answer_mask}

    def __len__(self):
        return len(self.dialogs)

    def build_combination(self, questions, answers):
        combis = []
        for i in range(len(questions)):
            history = []
            for hi_i, (question, answer) in enumerate(zip(questions[:i + 1], answers[:i] + [None])):
                history.append(question + " $u t%d" % (hi_i * 2))
                if answer is not None:
                    history.append(answer + " $s t%d" % (hi_i * 2 + 1))
            combis.append((history, questions[i], answers[i]))
        return combis

    def build_kb_vocab(self, questions):
        kbs_source = {}
        for i, question in enumerate(questions):
            for j, word in enumerate(question.split()):
                if word not in kbs_source:
                    kbs_source[word] = i * self.question_seq_len + j
        return kbs_source

    def get_data(self, file_path):
        dialogs = []
        with open(file_path, "r") as f:
            question = []
            answer = []
            for line in f:
                if line == "\n":
                    dialogs.extend(self.build_combination(question, answer))
                    question.clear(), answer.clear()
                else:
                    line = line[2:-1].split("\t")
                    question.append(line[0]), answer.append(line[1])
        return dialogs
