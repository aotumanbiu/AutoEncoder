# encoding: utf-8
import torch
import shutil
import os
import argparse
from tensorboardX import SummaryWriter
import torch.nn as nn
from src.models import AutoEncoder
from src.data import AnimeData
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from src.warmup import create_lr_scheduler
from torchvision.utils import make_grid
from tqdm import tqdm


def train(args):
    # ---------------------------------------------------------- #
    # 判断当前目录是否存在 logs 文件夹
    # ---------------------------------------------------------- #
    shutil.rmtree("logs") if os.path.isdir("logs") else ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------------------- #
    # 加载数据
    # ------------------------------------------------------------------------------- #
    Transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

    train_data = AnimeData(dataRoot=args.dataSet, subFold="train", transform=Transform)
    # train_data = datasets.ImageFolder(root="./data", transform=Transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # dataset / bathsize
    lenght = len(train_loader)

    # ------------------------------------------------------------ #
    # 初始化模型, 以及相应的参数优化器
    # ------------------------------------------------------------ #
    if args.cuda:
        autoencoder = AutoEncoder().to(device)
    else:
        autoencoder = AutoEncoder()

    optimizer = optim.Adam(params=autoencoder.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    criterion = nn.MSELoss()

    # --------------------------------------------------------- #
    # 是否采用 Warmup 学习率调整测率
    # --------------------------------------------------------- #
    if args.warmup:
        lr_scheduler = create_lr_scheduler(optimizer=optimizer,
                                           num_step=lenght,
                                           epochs=args.epoch,
                                           warmup_epochs=10)

    # --------------------------------------------------------- #
    # 是否采用 Tensorboard 可视化训练过程
    # --------------------------------------------------------- #
    if args.tensorboard:
        writer = SummaryWriter(logdir="logs")

    # -------------------------------------------------------------------- #
    # 开始训练
    # -------------------------------------------------------------------- #
    print("开始训练！！！！！！！")
    step = 0
    autoencoder.train()
    for epoch in range(args.epoch):

        tqdm_iter = tqdm(enumerate(train_loader), total=lenght)
        tqdm_iter.set_description(f"Epoch [{epoch + 1}/{args.epoch}]")

        total_loss = 0
        for idx, data in tqdm_iter:
            with torch.no_grad():
                data = data.to(device) if args.cuda else data

            out = autoencoder(data)
            optimizer.zero_grad()
            loss = criterion(data, out)
            loss.backward()
            optimizer.step()

            # ----------------------------------- #
            # 计算总损失
            # ----------------------------------- #
            total_loss += loss.item()

            # ----------------------------------- #
            # 更新学习率
            # ----------------------------------- #
            if args.warmup:
                lr_scheduler.step()

            # -------------------------------------------------------------- #
            # 显示详细信息
            # -------------------------------------------------------------- #
            tqdm_iter.set_postfix(**{'loss': loss.item(),
                                     'lr': optimizer.param_groups[0]['lr']})

            # -------------------------------------------------------------------------------- #
            # 可视化训练过程
            # -------------------------------------------------------------------------------- #
            if args.tensorboard:
                writer.add_scalar("Loss", loss.item(), epoch * lenght + idx)
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch * lenght + idx)

                img_grid_real = make_grid(data, normalize=True)
                img_grid_out = make_grid(out, normalize=True)
                writer.add_image("Real", img_grid_real, global_step=step)
                writer.add_image("Out", img_grid_out, global_step=step)
                step += 1

        if (epoch + 1) % 10 == 0:
            if not os.path.exists("./weigths"):
                os.mkdir("./weigths")
            print('Saving state, iter:', str(epoch + 1))
            torch.save(autoencoder.state_dict(), './weigths/Epoch%d-Total_Loss%.4f.pth' %
                       ((epoch + 1), total_loss))

    if args.tensorboard:
        writer.close()

    print("训练结束！！！！！！！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoEncoder")

    parser.add_argument("--cuda", metavar="True", type=bool, default=True,
                        help="Whether To Use GPU")
    parser.add_argument('--tensorboard', metavar="True", type=bool, default=True,
                        help='Tensorboard Integration')
    parser.add_argument('--batch_size', type=int, default=16, help="Batch_Size")
    parser.add_argument("--dataSet", type=str, default="./data", help="Dataset storage directory")
    parser.add_argument('--epoch', type=int, default=100, help="Total number of training")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="Adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam: decay of first order momentum of gradient")
    parser.add_argument("--warmup", type=bool, default=True, help="Whether to use warmup to adjust the learning rate")
    parser.add_argument('--load_checkpoints', metavar="False", type=bool, default=False,
                        help="Whether to pre-train")

    args = parser.parse_args()

    # 开始训练
    train(args)
