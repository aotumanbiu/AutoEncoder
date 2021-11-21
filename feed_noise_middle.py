import torch
from collections import OrderedDict
from src.vgg13 import Decoder
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

device = torch.device("cuda")
writer = SummaryWriter(logdir="logs")

# ---------------------------------------------- #
# 随机初始化噪声
# ---------------------------------------------- #
inputs = torch.rand((256, 512, 7, 7)).to(device)
data = DataLoader(inputs, batch_size=8)

# ---------------------------------------------- #
# 初始化解码器部分 Decoder
# ---------------------------------------------- #
decoder = Decoder().to(device)

# ------------------------------------------------------------------------------- #
# 加载权重
# ------------------------------------------------------------------------------- #
weigths = torch.load("Epoch100-Sigmoid-Total_Loss0.4214.pth", map_location=device)
new_state = OrderedDict()
for k, v in weigths.items():
    if "decoder" in k:
        name = k[8:]
        new_state[name] = v

decoder.load_state_dict(new_state)

# ------------------------------------------------------------------------------- #
# 测试
# ------------------------------------------------------------------------------- #
decoder.eval()
for idx, dt in enumerate(data):
    outs = decoder(dt)
    writer.add_images("out", outs, global_step=idx + 1)

writer.close()
