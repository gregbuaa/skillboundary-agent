import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 10
batch_size = 128

dataset = "mooc"  # mooc, junyi
ratio = 0.5

if dataset == "mooc":
    exer_n = 2704  # problem number
    knowledge_n = 6989  # skill number
    student_n = 5159  # student number
else:
    exer_n = 835  # problem number
    knowledge_n = 835  # skill number
    student_n = 10000  # student number
