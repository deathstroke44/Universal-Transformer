import os

for filename in os.listdir("babi-qa"):
    if filename.startswith("qa"):
        file_type = "train" if filename.find("train") > 0 else "test"
        task_num = filename.split("_")[0][2:]
        os.rename("babi-qa/" + filename, "babi-qa/task%s_%s.txt" % (task_num, file_type))
