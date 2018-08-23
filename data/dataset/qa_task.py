from torch.utils.data import Dataset


class BabiQADataset(Dataset):
    def __init__(self, path, enc_vocab, dec_vocab):
        self.path = path
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab
        self.data = self.get_dialog(path)

    def get_dialog(self, path):
        lines = open(path, "r", encoding="utf-8").readlines()
        dialog = []
        qa = []
        for i, line in enumerate(lines):
            if line == "\n" or (line[:2] == "1 " and i != 0):
                qa.extend(self.separate_dialog(dialog))
                dialog.clear()
            else:
                line = " ".join(line[:-1].split(" ")[1:])
                dialog.append(line)

        return qa

    def separate_dialog(self, dialog):
        story_history = []
        qa = []
        for line in dialog:
            if line.find("\t") >= 0:
                line = line.split("\t")
                qa.append((story_history.copy(), line[0], line[1]))
            else:
                story_history.append(line)
        return qa
