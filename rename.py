import os

types = {
    "trn": "train",
    "tst": "test",
    "tst-OOV": "test_oov",
    "dev": "dev",
    "candidates": "candidates"
}

for file in os.listdir("babi-dialog/"):
    if file.find("dialog-babi-task") == 0:
        task = file.split("-")[2]
        file_type = file.replace(".txt", "").split("-")
        file_type = file_type[-1] if file_type[-1] != "OOV" else "-".join(file_type[-2:])
        file_type = types[file_type]

        origin_path = "babi-dialog/%s" % file
        rename_path = "babi-dialog/%s_%s.txt" % (task, file_type)
        print(file, task, file_type)
        os.rename(origin_path, rename_path)
