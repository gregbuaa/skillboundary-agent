from typing import List

import torch

from cdm.config import device
from cdm.rcd.model import Net
from cdm.rcd.utils import CommonArgParser, construct_local_map

args = CommonArgParser().parse_args()
local_map = construct_local_map(args)
net = Net(args, local_map).to(device)


def get_stu_emb(stu_id: int) -> List[int]:
    """return skill array of student[stu_id]"""
    stu_tensor = torch.LongTensor([stu_id]).to(device)
    with torch.no_grad():
        stu_emb = torch.sigmoid(net.student_emb(stu_tensor))
    return stu_emb.squeeze().tolist()
