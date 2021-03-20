import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_relu_norm(
    in_channel, out_channels, kernel_size, stride, padding, relu=True, norm=True
):
    layers = [
        nn.Conv2d(
            in_channel, out_channels, kernel_size, stride, padding, bias=not norm,
        )
    ]
    if norm:
        layers += [nn.BatchNorm2d(out_channels)]
    if relu:
        layers += [nn.ReLU()]

    return nn.Sequential(*layers)


class OCRModel(nn.Module):
    def __init__(self, num_classes=19, dropout=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.CNN_Model = nn.Sequential(
            conv_relu_norm(3, 128, (3, 6), 1, (1, 1), True, False),
            nn.MaxPool2d(kernel_size=(2, 2)),
            conv_relu_norm(128, 64, (3, 6), 1, (1, 1), True, False),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        # 64 * 18 = 1152
        self.linear_1 = nn.Linear(1152, 64)
        self.dropout = nn.Dropout(dropout)

        self.GRU = nn.GRU(
            64, 32, num_layers=2, dropout=0.25, bidirectional=True, batch_first=True
        )  # batch_first might need to be True
        self.linear_2 = nn.Linear(
            32 * 2, num_classes + 1
        )  # multiply by two because we're using bidirectional GRU
        # plus 1 because there's an UNK token

    def forward(self, images, targets=None):
        batch_size = images.size(0)
        x = self.CNN_Model(images)  # shape: (N, 64, 18, 72)
        x = x.permute(0, 3, 1, 2).contiguous()  # shape: (N, 72, 64, 18)
        x = x.view(batch_size, x.size(1), -1)  # shape: (N, 72, 1152)
        x = F.relu(self.linear_1(x))  # shape: (N, 72, 64)
        x = self.dropout(x)  # shape: (N, 72, 64)
        x, _ = self.GRU(x)  # shape: (N, 72, 32 * 2)
        x = self.linear_2(x)  # shape: (N, 72, 20)
        x = x.permute(1, 0, 2).contiguous()  # shape: (72, N, 20)

        if targets is not None:
            log_softmaxed = F.log_softmax(x, dim=2)
            input_lens = torch.full(
                size=(batch_size,), fill_value=x.size(0), dtype=torch.int32
            )
            target_lens = torch.full(
                size=(batch_size,), fill_value=targets.size(1), dtype=torch.int32
            )
            loss_function = nn.CTCLoss(blank=0)
            loss = loss_function(log_softmaxed, targets, input_lens, target_lens)
            return x, loss
        return x, None


if __name__ == "__main__":
    images = torch.randn(32, 3, 75, 300)
    targets = torch.randint(0, 20, (32, 5))
    model = OCRModel()
    out = model(images=images, targets=targets)
    print(out[0].shape, out[1])
