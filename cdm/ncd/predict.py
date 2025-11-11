import json
from typing import List

from cdm.config import *
from model import Net

# init model
net = Net(student_n, exer_n, knowledge_n)
if dataset == "mooc":
    with open('cdm/mooc_epoch3_train0.7', 'rb') as f:
        net.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
else:
    with open('cdm/junyi_epoch4_train0.7', 'rb') as f:
        net.load_state_dict(torch.load(f, map_location=lambda s, loc: s))

net = net.to(device)
net.eval()

# load data
with open("data/uid.json", 'r', encoding='utf8') as f:
    uid2idx = json.load(f)


def get_stu_emb(stu_id: str) -> List[int]:
    """return skill array of student[stu_id]"""
    stu_id_idx = uid2idx[stu_id]
    stu_tensor = torch.LongTensor([stu_id_idx]).to(device)
    with torch.no_grad():
        stu_emb = torch.sigmoid(net.student_emb(stu_tensor))
    return stu_emb.squeeze().tolist()
