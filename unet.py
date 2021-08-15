# Sourced from: https://amaarora.github.io/2020/09/13/unet.html

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import hashlib
import PIL
import numpy as np

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(3, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64),
                 final_channels=3, out_size=None):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], final_channels, 1)
        self.out_size = out_size

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.out_size is not None:
            out = F.interpolate(out, self.out_size)
        return out

def convert_string_to_float(input_string):
    return float('.' + str(int(hashlib.md5(input_string.encode("utf8")).digest().hex(), 16))[-5:])

def load_file_as_tensor(file_path):
    return (torch.from_numpy(np.array(PIL.Image.open(file_path))) / 255).permute(2,0,1)

class RandomCanonicalDataset(Dataset):
    def __init__(self, root, random_set='randomized', canonical_set='canonical', datatype='all'):
        random_folder = os.path.join(root, random_set)
        canonical_folder = os.path.join(root, canonical_set)
        random_files = os.listdir(random_folder)
        canonical_files = os.listdir(canonical_folder)
        overlap = set(random_files).intersection(canonical_files)

        bounds_dict = {
            'all': (0, 1),
            'training': (0, 0.7),
            'validation': (0.7, 0.9),
            'testing': (0.9, 1.0),
        }

        bounds = bounds_dict[datatype]

        process = []
        for file in overlap:
            if bounds[0] <= convert_string_to_float(file) < bounds[1]:
                process.append(file)

        self.images = []
        print('Loading {} images'.format(len(process)))
        for i, file in enumerate(process):

            if not (i + 1) % 100:
                print('{} / {}'.format(i, len(process)))

            item = {
                'random': load_file_as_tensor(os.path.join(random_folder, file)),
                'canonical': load_file_as_tensor(os.path.join(canonical_folder, file)),
            }
            self.images.append(item)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i]

def write_tensor_to_rgb(tensor, file_path):

    if tensor.shape[0] == 3:
        tensor = tensor.permute(1,2,0)

    array = tensor.numpy()
    array[array > 1] = 1
    array[array < 0] = 0
    array = (array * 255).astype(np.uint8)
    img = PIL.Image.fromarray(array)
    img.save(file_path)


def run_eval(model, dataloader, loss_func, output_folder=None):
    device = torch.device("cuda:0")
    model.eval()
    counter = 0
    img_counter = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            inputs = data['random'].to(device)
            targets = data['canonical'].to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            running_loss += loss.item()

            counter += outputs.shape[0]

            if output_folder is not None:
                for output_tensor in outputs:
                    write_tensor_to_rgb(output_tensor.cpu(), os.path.join(output_folder, f'{img_counter}.png'))
                    img_counter += 1


    return running_loss, counter

if __name__ == '__main__':

    root = os.path.join(os.path.expanduser('~'), 'Documents', 'TrainingImages')
    train = True
    BATCH_SIZE = 4
    EVAL_FREQ = 2000
    PATIENCE = 25

    device = torch.device("cuda:0")
    net = UNet(out_size=(240, 424)).to(device)
    loss_func = nn.MSELoss()


    if train:
        train_dataset = RandomCanonicalDataset(root, datatype='training')
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        valid_dataset = RandomCanonicalDataset(root, datatype='validation')
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        optimizer = optim.Adam(net.parameters(), lr=0.0001)


        eval_counter = 0
        best_loss = np.inf
        losing_streak = 0
        is_done = False
        all_losses = []

        for epoch in range(100):

            if is_done:
                break

            running_loss = 0.0
            for i, data in enumerate(train_loader):

                print('{} / {}'.format((i + 1) * BATCH_SIZE, len(train_dataset)))

                net.train()
                inputs = data['random'].to(device)
                targets = data['canonical'].to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = loss_func(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                eval_counter += inputs.shape[0]
                if eval_counter >= EVAL_FREQ:

                    eval_counter -= EVAL_FREQ
                    print('=== RUNNING EVALUATION ===')

                    total_loss, n = run_eval(net, valid_loader, loss_func)
                    avg_loss = total_loss / n
                    all_losses.append(avg_loss)


                    print('=== CURRENT LOSS: {:.4f}'.format(avg_loss))
                    print('=== BEST LOSS: {:.4f}'.format(best_loss))

                    if avg_loss < best_loss:
                        print('=== Best loss improved, saving model!')
                        best_loss = avg_loss
                        torch.save(net.state_dict(), 'unet_best.model')
                        print('=== Best model saved')
                        losing_streak = 0
                    else:
                        losing_streak += 1
                        print('=== Loss has not improved after {} evaluations...'.format(losing_streak))
                        if losing_streak >= PATIENCE:
                            print('=== Loss has exceeded patience threshold, stopping training!')
                            is_done = True
                            break

            print('Finished Epoch {}!'.format(epoch + 1))
            np.save('losses.npy', np.array(all_losses))

    else:
        net.load_state_dict(torch.load('unet_best.model'))
        testing_dataset = RandomCanonicalDataset(root, datatype='testing')
        testing_loader = DataLoader(testing_dataset, batch_size=4, shuffle=False, num_workers=2)
        run_eval(net, testing_loader, loss_func, output_folder=os.path.join(root, 'predictions'))

