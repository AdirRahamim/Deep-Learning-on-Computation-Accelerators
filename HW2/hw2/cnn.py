import torch
import itertools as it
import torch.nn as nn


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes: int, channels: list,
                 pool_every: int, hidden_dims: list):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ReLU)*P -> MaxPool]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs should exist at the end, without a MaxPool after them.
        # ====== YOUR CODE: ======
        P = self.pool_every
        N = len(self.channels)
        num_conv = 1

        for in_c, out_c in zip([in_channels] + self.channels, self.channels):
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=(3, 3), padding=1))
            layers.append(nn.ReLU())
            if not num_conv % P:
                layers.append(nn.MaxPool2d(2))
            num_conv += 1

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        #  (Linear -> ReLU)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        N = len(self.channels)
        P = self.pool_every
        in_features = self.channels[-1] * (in_w // (2 ** (N // P))) * (in_h // (2 ** (N // P)))

        for l_in, l_out in zip([in_features] + self.hidden_dims, self.hidden_dims):
            layers.append(nn.Linear(l_in, l_out))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list,
                 batchnorm=False, dropout=0.):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
        convolution in the block. The length determines the number of
        convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
        be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
        convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
        Zero means don't apply dropout.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order). Should end with a
        #    final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use. This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        main_path, shortcut_path = [], []
        N = len(channels)

        idx = 0
        for in_c, out_c in zip([in_channels] + channels, channels):
            main_path.append(nn.Conv2d(in_c, out_c, kernel_size=kernel_sizes[idx], padding=kernel_sizes[idx] // 2))
            if dropout and idx < N - 1:
                main_path.append(nn.Dropout2d(p=dropout))

            if batchnorm and idx < N - 1:
                main_path.append(nn.BatchNorm2d(out_c))

            if idx < N - 1:
                main_path.append(nn.ReLU())

            idx += 1

        self.main_path = nn.Sequential(*main_path)

        if in_channels != channels[-1]:
            shortcut_path.append(nn.Conv2d(in_channels, channels[-1], kernel_size=1, bias=False))

        self.shortcut_path = nn.Sequential(*shortcut_path)

        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ReLU)*P -> MaxPool]*(N/P)
        #   \------- SKIP ------/
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ReLUs (with a skip over them) should exist at the end,
        #    without a MaxPool after them.
        #  - Use your ResidualBlock implemetation.
        # ====== YOUR CODE: ======
        P = self.pool_every
        N = len(self.channels)
        idx = 1

        curr_channels = []
        residual_input = in_channels

        for conv_dim in self.channels:
            curr_channels.append(conv_dim)
            if not idx % P:
                layers.append(ResidualBlock(residual_input, curr_channels, [3] * P))
                layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
                residual_input = conv_dim
                curr_channels = []
            idx += 1

        if N % P:
            layers.append(ResidualBlock(residual_input, curr_channels, [3] * len(curr_channels)))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

#====New calsses====

class CustomConv2d(nn.Module):
    """
    Based on insection model(as proposed) - make 1x1, 3x3 and 5x5 convolution and return average of them
    """
    def __init__(self, in_channels, out_channels, ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(5,5), padding=2)
        self.bath_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = (1/3) * (out1 + out2 + out3)
        out = self.bath_norm(out)
        return out


class CustomResidualBlock(nn.Module):

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list,
                 batchnorm=True, dropout=0.2):
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        self.main_path, self.shortcut_path = None, None

        main_path, shortcut_path = [], []
        N = len(channels)

        idx = 0
        for in_c, out_c in zip([in_channels] + channels, channels):
            main_path.append(CustomConv2d(in_c, out_c))
            if dropout and idx < N - 1:
                main_path.append(nn.Dropout2d(p=dropout))

            if batchnorm and idx < N - 1:
                main_path.append(nn.BatchNorm2d(out_c))

            if idx < N - 1:
                main_path.append(nn.ReLU())

            idx += 1

        self.main_path = nn.Sequential(*main_path)

        if in_channels != channels[-1]:
            shortcut_path.append(nn.Conv2d(in_channels, channels[-1], kernel_size=1, bias=False))

        self.shortcut_path = nn.Sequential(*shortcut_path)

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        P = self.pool_every
        N = len(self.channels)
        idx = 1

        curr_channels = []
        residual_input = in_channels

        for conv_dim in self.channels:
            curr_channels.append(conv_dim)
            if not idx % P:
                layers.append(CustomResidualBlock(residual_input, curr_channels, [3] * P))
                layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
                residual_input = conv_dim
                curr_channels = []
            idx += 1

        if N % P:
            layers.append(CustomResidualBlock(residual_input, curr_channels, [3] * len(curr_channels)))

        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []

        N = len(self.channels)
        P = self.pool_every
        in_features = self.channels[-1] * (in_w // (2 ** (N // P))) * (in_h // (2 ** (N // P)))

        for l_in, l_out in zip([in_features] + self.hidden_dims, self.hidden_dims):
            layers.append(nn.Linear(l_in, l_out))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))

        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):

        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out
    # ========================
