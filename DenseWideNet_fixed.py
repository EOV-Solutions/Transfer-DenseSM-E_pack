import torch
from torch import nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, in_channel, width):
        super().__init__()

        self.linear1 = nn.Linear(in_channel, width)
        self.linear2 = nn.Linear(width, width)
        self.linear3 = nn.Linear(width, width)
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)
        self.bn3 = nn.BatchNorm1d(width)

    def forward(self, x):
        out = F.relu(self.bn1(self.linear1(x)))
        out = F.relu(self.bn2(self.linear2(out)))
        out = F.relu(self.bn3(self.linear3(out)))
        out = torch.cat((out, x), dim=1)
        return out


class SingleFcBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.linear1 = nn.Linear(in_channel, out_channel)
        self.bn1 = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        out = self.bn1(F.relu(self.linear1(x)))
        return out


class ConvBlock(nn.Module):

    def __init__(self, in_channel_ts):
        super().__init__()
        self.window_size = [2, 2, 4]
        filter_size = [32, 16, 4]
        self.conv1 = nn.Conv1d(in_channel_ts, filter_size[0], kernel_size=5, padding=2, padding_mode='replicate')
        self.bn1 = nn.BatchNorm1d(filter_size[0])
        self.conv2 = nn.Conv1d(filter_size[0], filter_size[1], kernel_size=5, padding=2, padding_mode='replicate')
        self.bn2 = nn.BatchNorm1d(filter_size[1])
        self.conv3 = nn.Conv1d(filter_size[1], filter_size[2], kernel_size=5, padding=2, padding_mode='replicate')
        self.bn3 = nn.BatchNorm1d(filter_size[2])
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)
        self.dp3 = nn.Dropout(0.5)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.avg_pool1d(self.dp1(out), self.window_size[0])
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool1d(self.dp2(out), self.window_size[1])
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.avg_pool1d(self.dp3(out), self.window_size[2], ceil_mode=True)
        out = out.view(out.size(0), -1)
        return out
    

class TCNEncoder(nn.Module):

    def __init__(self, in_channel_ts, out_dim = 12, hidden_dim = 64):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channel_ts, hidden_dim, kernel_size=3, padding = 2, dilation = 2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size = 3, padding = 4, dilation = 4),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_dim, kernel_size = 3, padding = 8, dilation = 8),
            nn.AdaptiveAvgPool1d(1) # [B, out_dim, 1]
        )

    def forward(self, x): # [B, 3, 46]
        x = self.tcn(x) # [B, out_dim, 1]
        return x.squeeze(-1)



class DWN(nn.Module):
    def __init__(self, width, block_num, fc_num=1, fc_channel=32):
        self.width = width
        # If you want to cut the position of the data, you can change the number of features here: in_channel = 15 + 12
        self.in_channel = 18 + 12
        self.in_ts_fe = 3
        self.block_num = block_num
        # self.block = block
        self.fc_channel = fc_channel
        self.fc_num = fc_num
        self.fc_in = block_num * width + self.in_channel

        super().__init__()
        # Convolutional block processing time-series data (46 x 3)
        # self.conv_block = ConvBlock(self.in_ts_fe)
        self.tcn_encoder = TCNEncoder(self.in_ts_fe)

        self.norm = nn.BatchNorm1d(self.in_channel)

        self.dense_layers = self._make_layer(DenseBlock, self.in_channel, self.width)

        self.fc_layers = self._make_layer_fc(SingleFcBlock, self.fc_in, self.fc_channel)

        self.out = nn.Linear(self.fc_channel, 1)
        '''
        self.fc_end = nn.Sequential(
            nn.Linear(block_num * width + self.in_channel, fn_final),
            nn.ReLU(),
            nn.BatchNorm1d(fn_final),
            nn.Linear(fn_final, 1)
            )
        '''

    def _make_layer(self, block, in_channel, width):
        layers = []
        for bk in range(self.block_num):
            temp_in = bk * width + in_channel
            layers.append(block(temp_in, width))
        return nn.Sequential(*layers)

    def _make_layer_fc(self, block, in_channel, out_channel):
        layers = []
        for bk in range(self.fc_num):
            if bk > 0:
                in_channel = out_channel
            layers.append(block(in_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        print("input shape",x.shape)
        x = torch.tensor_split(x, [18, 157], dim=1)
        x1, x2 = x[0], x[1]
        x2 = x2.reshape(x2.shape[0], 3, 46)
        # x2 = self.conv_block(x2)
        x2 = self.tcn_encoder(x2)
        out = torch.cat((x1, x2), dim=1)
        out = self.norm(out)
        out = self.dense_layers(out)
        out = self.fc_layers(out)
        out = self.out(out)
        return out


class DWN_feature(nn.Module):
    def __init__(self, width, block_num, fc_num=1, fc_channel=32):
        self.width = width
        self.in_channel = 15 + 12 # 12 is the number of features from time series data. 18 = 15 one-shote data + 3 last day's data
        self.in_ts_fe = 3
        self.block_num = block_num
        # self.block = block
        self.fc_channel = fc_channel
        self.fc_num = fc_num
        self.fc_in = block_num * width + self.in_channel

        super().__init__()

        self.conv_block = ConvBlock(self.in_ts_fe)
        self.tcn_encoder = TCNEncoder(self.in_ts_fe)

        self.norm = nn.BatchNorm1d(self.in_channel)

        self.dense_layers = self._make_layer(DenseBlock, self.in_channel, self.width)

        self.fc_layers = self._make_layer_fc(SingleFcBlock, self.fc_in, self.fc_channel)

    def _make_layer(self, block, in_channel, width):
        layers = []
        for bk in range(self.block_num):
            temp_in = bk * width + in_channel
            layers.append(block(temp_in, width))
        return nn.Sequential(*layers)

    def _make_layer_fc(self, block, in_channel, out_channel):
        layers = []
        for bk in range(self.fc_num):
            if bk > 0:
                in_channel = out_channel
            layers.append(block(in_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        # print("input shape",x.shape)
        # x = torch.tensor_split(x, [18, 157], dim=1)
        # x1, x2 = x[0], x[1]
        # x2 = x2.reshape(x2.shape[0], 3, 46)
        # x2 = self.conv_block(x2)
        # out = torch.cat((x1, x2), dim=1)
        x = torch.tensor_split(x, [12, 15, 18, 157], dim = 1)
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        x4 = x4.reshape(x4.shape[0], 3, 46)
        x4 = self.conv_block(x4)
        out = torch.cat((x1, x3, x4), dim=1)
        out = self.norm(out)
        out = self.dense_layers(out)
        out = self.fc_layers(out)
        return out
