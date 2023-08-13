# Author: Luca Trautmann
# Description: Implementation of the Deep Griffin Lim model

# Load libraries
import torch 
import torch.nn as nn

# Define the GLU (Gated Linear Unit) class
class convGLU(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, kernel_size=(7,7), padding='same', batchnorm=False):
        super().__init__()
        if padding == 'same':
            padding = (kernel_size[0]//2, kernel_size[1]//2)
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, padding=padding)  # 2D convolutional layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function
        if batchnorm:
            self.conv = nn.Sequential(
                self.conv,
                nn.BatchNorm2d(out_channels * 2)  # Batch normalization layer
            )

    def forward(self, x):
        x = self.conv(x)  # Apply convolutional layer
        channel = x.shape[1]  # Get the number of channels

        x = x[:, :channel//2, :, :] * self.sigmoid(x[:, channel//2:, :, :])  # Apply GLU (Gated Linear Unit)
        return x


# Define the DNN (Deep Neural Network) class
class DNN(nn.Module):
    def __init__(self,padding=None,additional_conv=False):
        super().__init__()
        self._hidden_channels = 32
        self.initial = nn.Sequential(
            nn.Conv2d(7, self._hidden_channels, (11,11), padding=padding) if additional_conv else nn.Identity(), # in_channel = 6 because we concatenate the real and imag part of the complex spectrogram
            convGLU(self._hidden_channels if additional_conv else 7, self._hidden_channels, (11,11), padding='same'))

        self.mid = nn.Sequential(
            nn.Conv2d(self._hidden_channels, self._hidden_channels, (7,3), padding=(7//2, 3//2)) if additional_conv else nn.Identity(),
            convGLU(self._hidden_channels, self._hidden_channels, (7,3), padding='same'),
            nn.Conv2d(self._hidden_channels, self._hidden_channels, (7,3), padding=(7//2, 3//2)) if additional_conv else nn.Identity(),
            convGLU(self._hidden_channels, self._hidden_channels, (7,3), padding='same'),
        )

        self.final = nn.Sequential(
            nn.Conv2d(self._hidden_channels, 1, (7,3), padding=(7//2, 3//2)) if additional_conv else nn.Identity(),
            convGLU(self._hidden_channels,self._hidden_channels, (7,3), padding='same'),
            nn.Conv2d(self._hidden_channels, 2, (7,3), padding=(7//2, 3//2)),
        )

    def forward(self, x):
        x = self.initial(x)
        residual = x
        x = self.mid(x)
        x += residual
        x = self.final(x)
        return x


# Define the Deep Griffin Lim class
class DeepGriffinLim(nn.Module):
    """
    Deep Griffin Lim class for audio signal reconstruction.
    This class uses a deep neural network (DNN) to perform the Griffin-Lim algorithm for audio signal reconstruction.
    """
    def __init__(self,blocks=10, n_fft=1024, hop_size=512, win_size=1024, window='hann_window'):
        """
        Initialize the DeepGriffinLim class.
        Args:
            blocks (int): Number of DNN blocks.
            n_fft (int): FFT size.
            hop_size (int): Hop size for STFT.
            win_size (int): Window size for STFT.
            window (str): Window type for STFT.
        """
        super().__init__()
        self.dnn_blocks = nn.ModuleList([DNN() for _ in range(blocks)]) # Initialize DNN blocks

    def stft(self, x, n_fft=1024, hop_size=512, win_size=1024):
        """
        Perform Short-Time Fourier Transform (STFT) on the input signal.
        Args:
            x (Tensor): Input signal.
            n_fft (int): FFT size.
            hop_size (int): Hop size.
            win_size (int): Window size.
        Returns:
            Tensor: STFT of the input signal.
        """
        return torch.stft(x, n_fft=n_fft, hop_length=hop_size, win_length=win_size, return_complex=True)

    def istft(self, x, n_fft=1024, hop_size=512, win_size=1024):
        """
        Perform Inverse Short-Time Fourier Transform (ISTFT) on the input signal.
        Args:
            x (Tensor): Input signal.
            n_fft (int): FFT size.
            hop_size (int): Hop size.
            win_size (int): Window size.
        Returns:
            Tensor: ISTFT of the input signal.
        """
        return torch.istft(x, n_fft=n_fft, hop_length=hop_size, win_length=win_size)

    def magswap(self, mag, x_tilda):
        """
        Perform magnitude swapping on the input signal.
        Args:
            mag (Tensor): Magnitude of the signal.
            x_tilda (Tensor): Input signal.
        Returns:
            Tensor: Signal after magnitude swapping.
        """
        return mag * x_tilda / torch.abs(x_tilda)

    def forward(self,x_tilda, mag, added_depth=1):
        """
        Forward pass of the DeepGriffinLim model.
        Args:
            x_tilda (Tensor): Input signal.
            mag (Tensor): Magnitude of the signal.
            added_depth (int): Additional depth for the DNN.
        Returns:
            Tuple[Tensor]: Output of the forward pass.
        """
        subblock_out = []
        for _ in range(added_depth):
            for subblock in self.dnn_blocks:
                # Perform magnitude swapping and STFT on the input signal
                y_tilda = self.magswap(mag=mag,x_tilda=x_tilda)
                z_tilda = self.stft(self.istft(y_tilda.squeeze(1)))

                # Transform the input signal to float and concatenate with the magnitude
                dnn_in = self.transform_to_float([x_tilda, y_tilda, z_tilda.unsqueeze(1)])
                dnn_in = torch.cat([dnn_in, mag], dim=1)

                # Perform forward pass of the DNN
                dnn_out = subblock(dnn_in)
                residual  = torch.complex(dnn_out[:,0,...], dnn_out[:,1,...])

                # Update the input signal for the next iteration
                x_tilda = (z_tilda - residual).unsqueeze_(1)
                subblock_out.append(residual)

        # Perform final magnitude swapping on the input signal
        final = self.magswap(mag=mag,x_tilda=x_tilda)
        return z_tilda.unsqueeze_(1), residual.unsqueeze_(1), final, subblock_out

    @staticmethod
    def transform_to_float(tensor_list: list):
        """
        Transform the input tensor list to float.
        Args:
            tensor_list (list): List of input tensors.
        Returns:
            Tensor: Transformed tensor.
        """
        output = []
        for idx, i in enumerate(range(len(tensor_list))):
            if tensor_list[i].dtype == torch.complex64 and tensor_list[i].dim() == 4:
                output.append(torch.cat([tensor_list[i].real, tensor_list[i].imag], dim=1))
            else:
                print(f'Input {idx} is not a complex tensor with 4 dimensions')
                return None

        return torch.cat(output, dim=1)

