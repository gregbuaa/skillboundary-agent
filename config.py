import torch

dataset_split_seed = 42
dataset_shuffle_seed = 1314
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = "mooc"  # "mooc", "junyi"
gpt_model = "gpt-4o"  # "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o"
ratio = 3
