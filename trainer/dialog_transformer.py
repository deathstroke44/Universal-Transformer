import torch
import torch.nn as nn
from torch.optim import Adam
import nsml


class UniversalTransformerQATrainer:
    def __init__(self, model, dataloader, device):
        self.criterion = nn.NLLLoss()
        self.optimizer = Adam(model.parameters())
        self.model = model
        self.device = device
        self.dataloader = dataloader

    def train(self, epoch, verbose=0):
        return self.trainer(self.dataloader["train"], epoch)

    def test(self, epochs, epoch_sample=1, verbose=0):
        self.model.train(False)
        output = self.trainer(self.dataloader["test"], epochs, train=False, log_code="test", epoch_sample=epoch_sample)
        self.model.train(True)
        return output

    def trainer(self, dataloader, epoch, train=True, log_code="train", epoch_sample=1, verbose=0):
        avg_loss, total_correct = 0.0, 0.0
        total_nelement = 0
        for step, data in enumerate(dataloader):
            story, answer = data["history"], data["answer"]
            story_mask, answer_mask = data["history_mask"], data["answer_mask"]

            # Forward tensor to GPU
            story, answer = story.to(self.device), answer.to(self.device)
            story_mask, answer_mask = story_mask.to(self.device), answer_mask.to(self.device)

            output = self.model.forward(story, answer if train else None, story_mask, answer_mask)
            loss = self.criterion(output.transpose(-1, 1), answer)

            output_word = output.exp().argmax(dim=-1)
            correct = output_word.eq(answer).sum().float()
            acc = correct / answer.nelement() * 100

            total_correct += correct.item()
            total_nelement += answer.nelement()
            avg_loss += loss.item()

            # if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            output_log = {
                "epoch": epoch,
                "step": step,
                "%s_loss" % log_code: loss.item(),
                "%s_acc" % log_code: acc.item()
            }

            if verbose == 0:
                print(output_log)

            if verbose <= 1:
                output_log["step"] = epoch / epoch_sample * len(dataloader) + step
                nsml.report(**output_log)

        avg_loss /= len(dataloader)
        avg_acc = total_correct / total_nelement * 100.0
        return avg_loss, avg_acc
