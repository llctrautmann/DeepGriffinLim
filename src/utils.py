import os
import random
import torch
import http.client, urllib
from dotenv import load_dotenv
from hyperparameter import hp
import numpy as np
import torchvision
import torchaudio
import matplotlib.pyplot as plt
import librosa
import wandb
# from IPython.display import Audio

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class HealthCheckDashboard:

    def __init__(self, train_loader, model, writer):
        self.train_loader = train_loader
        self.model = model
        self.writer = writer
        self.step = 0  # Just a placeholder value. Adjust as needed.

    def format_tensor_details(self, tensor):
        return f"dim: {str(list(tensor.shape)):<25} dtype: {tensor.dtype}"

    def display_header(self):
        print("+--------------------------------------------------------------------+")
        print("|                       HEALTH CHECK DASHBOARD                       |")
        print("+--------------------------------------------------------------------+")

    def display_tensor_details(self, name, tensor):
        print(f"| {name:<8} -> {self.format_tensor_details(tensor)}")

    def display_separator(self):
        print("+--------------------------------------------------------------------+")

    def perform_healthcheck(self):
        self.display_header()

        for i, batch in enumerate(self.train_loader):
            clear, noisy, mag, label = batch

            self.display_tensor_details("Clear", clear)
            self.display_tensor_details("Noisy", noisy)
            self.display_tensor_details("Mag", mag)

            try:
                clear_grid = torchvision.utils.make_grid(torch.angle(clear), padding=20)
                self.writer.add_images("Clear", clear_grid, dataformats='CHW', global_step=self.step)
            except Exception as e:
                print(f"| Error    -> {e}")

            self.model.train()
            z_tilda, residual, final, subblock_out = self.model(x_tilda=noisy, mag=mag)

            self.display_tensor_details("z_tilda", z_tilda)
            self.display_tensor_details("Residual", residual)
            self.display_tensor_details("Final", final)

            try:
                res_grid = torchvision.utils.make_grid(torch.angle(residual), padding=20)
                final_grid = torchvision.utils.make_grid(torch.angle(final), padding=20)
                z_tilda_grid = torchvision.utils.make_grid(torch.angle(z_tilda), padding=20)

                self.writer.add_images("z_tilda", z_tilda_grid, dataformats='CHW', global_step=self.step)
                self.writer.add_images("Final", final_grid, dataformats='CHW', global_step=self.step)
                self.writer.add_images("Residual", res_grid, dataformats='CHW', global_step=self.step)
            except Exception as e:
                print(f"| Error    -> {e}")

            self.display_separator()
            break

# Monitor the training loss
# Load .env file
load_dotenv()

# Get the API key
APP_TOKEN = os.getenv('APP_TOKEN')
USER_KEY = os.getenv('USER_KEY')

def send_push_notification(epoch, message):
    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request("POST", "/1/messages.json",
      urllib.parse.urlencode({
        "token": APP_TOKEN,
        "user": USER_KEY,
        "message": f"Epoch {epoch} completed: validation loss = {message}",
      }), { "Content-type": "application/x-www-form-urlencoded" })
    response = conn.getresponse()

@torch.no_grad()
def resize_signal_length(signal, signal_length):
    if signal.shape[-1] > signal_length:
        signal = signal[...,:signal_length]
        return signal

    elif signal.shape[-1] < signal_length:
        length_diff = signal_length - signal.shape[-1]
        prefix = torch.zeros((1,length_diff//2))
        suffix = torch.zeros((1,length_diff//2))
        signal = torch.cat([prefix,signal.reshape(1,-1),suffix],dim=1)

        if len(signal[-1]) == signal_length:
            return signal
        else:
            length_diff = signal_length - len(signal[-1])
            signal = torch.cat([signal,torch.zeros((1,length_diff))],dim=-1)
            return signal
    else:
        return signal

def visualize_tensor(clear, recon, key, loss, step):
    # Separate the real and imaginary parts of the complex tensor
    real = clear.abs()
    imag = clear.angle()
    recon = recon.angle()
    amptodb = librosa.amplitude_to_db

    # convert to numpy and detach from the graph
    real = real.detach().cpu().numpy()
    imag = imag.detach().cpu().numpy()
    recon = recon.detach().cpu().numpy()

    # Create a grid of subplots for real and imaginary parts
    fig, axs = plt.subplots(1, 3, figsize=(30, 5))

    # Plot the magnitude part, real part and reconstructed part
    titles = ['Magnitude', 'Ground Truth Phase', 'Reconstructed Phase']
    images = [amptodb(real), imag, recon]
    for i in range(3):
        im = librosa.display.specshow(images[i][0][0], ax=axs[i])
        axs[i].set_title(titles[i])
        fig.colorbar(im, ax=axs[i])

    # Save the plot to a file in a specified folder
    output_dir = os.path.join('..', 'out', str(loss), 'img')
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, f'im_{step}_{key}_type_{loss}.png')
    plt.savefig(img_path)
    plt.close()

    # Log the image file with weights and biases
    if hp.device.startswith(hp.wandb_device):
        wandb.log({f"{key}_type_{loss}": [wandb.Image(img_path, caption=f"{key}_type_{loss}")]}, step=step)



