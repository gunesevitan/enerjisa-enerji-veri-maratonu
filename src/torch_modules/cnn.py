import torch
import torch.nn as nn


class Conv1dBlock(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, stride, skip_connection=False):

        super(Conv1dBlock, self).__init__()

        self.skip_connection = skip_connection
        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=(kernel_size,),
                stride=(stride,),
                padding=(kernel_size // 2,),
                padding_mode='replicate',
                bias=True
            ),
            nn.BatchNorm1d(num_features=output_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.downsample = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=(1,),
                stride=(stride,),
                bias=False
            ),
            nn.BatchNorm1d(num_features=output_dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):

        output = self.conv_block(x)
        # Use resnet-like skip connections
        if self.skip_connection:
            x = self.downsample(x)
            output += x
        output = self.relu(output)

        return output


class ConvolutionalNeuralNetwork(nn.Module):

    def __init__(self, input_dim, sequence_length):

        super(ConvolutionalNeuralNetwork, self).__init__()

        self.conv_block1 = Conv1dBlock(
            input_dim=input_dim,
            output_dim=64,
            kernel_size=5,
            stride=1,
            skip_connection=True,
        )
        self.conv_block2 = Conv1dBlock(
            input_dim=64,
            output_dim=96,
            kernel_size=5,
            stride=1,
            skip_connection=True,
        )
        self.conv_block3 = Conv1dBlock(
            input_dim=96,
            output_dim=128,
            kernel_size=3,
            stride=1,
            skip_connection=True,
        )
        self.conv_block4 = Conv1dBlock(
            input_dim=128,
            output_dim=256,
            kernel_size=3,
            stride=1,
            skip_connection=True,
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(in_features=256, out_features=sequence_length, bias=True),
            nn.ReLU()
        )

    def forward(self, x):

        x = torch.transpose(x, 1, 2)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.pooling(x).view(-1, x.shape[1])
        output = self.head(x)

        return output.view(-1)
