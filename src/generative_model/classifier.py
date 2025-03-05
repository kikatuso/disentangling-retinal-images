import torch
import torchvision


class LinearClassifier(torch.nn.Module):
    """Linear classifier head.

    Attributes:
        w_shape: Number of latent space dimensions.
        c_shape: Number of classes.
    """

    def __init__(
        self,
        w_shape: int = 512,
        c_shape: int = 2,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(w_shape, c_shape)

    def forward(self, w):
        return self.linear(w)


class Classifier(torch.nn.Module):
    """MLP classifier (nonlinear).

    Attributes:
        hidden_layers: Number of hidden layers.
        w_shape: Number of latent space dimensions.
        c_shape: Number of classes.
    """

    def __init__(
        self,
        hidden_layers: int = 1,
        w_shape: int = 512,
        c_shape: int = 2,
    ):
        super().__init__()

        hidden_channels = [w_shape // (2 * (i + 1)) for i in range(hidden_layers)]

        self.classifier = torchvision.ops.MLP(
            in_channels=w_shape,
            hidden_channels=hidden_channels
            + [
                c_shape,
            ],
            activation_layer=torch.nn.ReLU,
        )

    def forward(self, w):
        return self.classifier(w)


class GRLayer(torch.autograd.Function):
    """Gradient reversal layer.

    Acts as am identity function in the forward pass and inverts the gradient during
    backpropagation.
    """

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.scale, None


class AdvClassifier(torch.nn.Module):
    """Nonlinear adversarial classification head with gradient reversal layer (GRL).

    Reference for GRL:
        paper: https://arxiv.org/abs/1505.07818
        example implementation: Adversarial classifier:
            https://github.com/NaJaeMin92/pytorch-DANN

    Attributes:
        hidden_layers: Number of hidden layers.
        z_shape: Latent space dimension.
        c_shape: Output/class dimension.
    """

    def __init__(
        self,
        hidden_layers: int = 1,
        z_shape: int = 512,
        c_shape: int = 2,
    ):
        super().__init__()

        hidden_channels = [z_shape // (2 * (i + 1)) for i in range(hidden_layers)]

        self.layers = torchvision.ops.MLP(
            in_channels=z_shape,
            hidden_channels=hidden_channels
            + [
                c_shape,
            ],
            activation_layer=torch.nn.ReLU,
        )

    def forward(self, z, alpha):
        reversed_input = GRLayer.apply(z, alpha)
        x = self.layers(reversed_input)
        return x
