import torch
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from src.vgg13 import AutoEncoder
from tensorboardX import SummaryWriter
from tqdm import tqdm

device = torch.device("cuda")
writer = SummaryWriter(logdir="logs")


inputs = torch.rand((256, 3, 224, 224)).to(device)
data = DataLoader(inputs, batch_size=8)

model_state_dict = torch.load("Epoch100-Sigmoid-Total_Loss0.4214.pth", map_location=device)
model = AutoEncoder()
model.load_state_dict(model_state_dict)


net = model.to(device)
net.eval()
tqdm_iter = tqdm(enumerate(data), total=len(data))
tqdm_iter.set_description("Test")
for idx, data in tqdm_iter:
    with torch.no_grad():
        inputs = data.to(device)
    outs = net(inputs)

    ins, out = make_grid(data), make_grid(outs)
    writer.add_image('inputs', ins, global_step=idx + 1)
    writer.add_image('outputs', out, global_step=idx + 1)

writer.close()
