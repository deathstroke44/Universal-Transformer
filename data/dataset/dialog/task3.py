from .task1 import Task1DataSet


class Task3DataSet(Task1DataSet):
    def get_data(self, file_path):
        dialogs = []
        with open(file_path, "r") as f:
            question = []
            answer = []
            kb = []
            for line in f:
                if line == "\n":
                    combis = self.build_combination(question, answer)
                    combis = [(kb + history, answer) for history, answer in combis]
                    dialogs.extend(combis)
                    question.clear(), answer.clear(), kb.clear()
                else:
                    line = line[line.find(" ") + 1:-1]
                    if line.find("\t") < 0:
                        kb.append(line)
                    else:
                        line = line.split("\t")
                        question.append(line[0]), answer.append(line[1])
        return dialogs
