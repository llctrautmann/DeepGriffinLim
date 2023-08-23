import os
import random
import torch
import http.client, urllib
from dotenv import load_dotenv
import librosa
import numpy as np
import torchvision
# import matplotlib.pyplot as plt
# from IPython.display import Audio


# def visualize_complex_tensor(tensor):
#     if tensor.isinstance(torch.Tensor):
#         tensor = tensor.detach()
#     else:
#         pass
#     # Separate the real and imaginary parts of the complex tensor
#     real = tensor.abs()
#     imag = tensor.angle()
#     amptodb = torchaudio.transforms.AmplitudeToDB()

#     # Create a grid of subplots for real and imaginary parts
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))

#     # Plot the real part
#     im1 = axs[0].imshow(amptodb(real), cmap='magma')
#     axs[0].set_title('Mag')

#     # Plot the imaginary part
#     im2 = axs[1].imshow(imag, cmap='Blues')
#     axs[1].set_title('Phase')

#     # Add color bar
#     fig.colorbar(im1, ax=axs[0])
#     fig.colorbar(im2, ax=axs[1])

#     # Show the plot
#     plt.show()


# def plot_spectrograms(batch: torch.Tensor,magnitude=True,width=10,height=3):
#     plt.figure(figsize=(width,height))
#     if magnitude:
#         librosa.display.specshow(batch[0][0].numpy(),
#                                 sr=48000,
#                                 x_axis='time',
#                                 y_axis='linear',
#                                 # vmin=0,
#                                 # vmax=44100//2
#                                 )
#         plt.colorbar(format="%+2.f")
#         plt.show()
#     else:
#         librosa.display.specshow(batch[0][1].numpy(),
#                                 sr=48000,
#                                 x_axis='time',
#                                 y_axis='linear',
#                                 cmap='Blues',
#                                 )
#         plt.colorbar(format="%+2.f")
#         plt.show()


# def play_tensor(wave: torch.tensor,sr=44100):
#     numpy_waveform = wave.numpy()
#     return Audio(numpy_waveform, rate=sr)


# def plot_to_tensorboard(writer, loss_critic, loss_gen,real,fake, tb_step,images=False):
#     writer.add_scalar('loss_critic', loss_critic, tb_step)
#     writer.add_scalar('loss_gen', loss_gen, tb_step)


#     if images:

#         with torch.no_grad():
#             img_grid_fake = torchvision.utils.make_grid(fake[:4], normalize=True) # added [1] so we plot the reconstructed phase not magnitude
#             img_grid_real = torchvision.utils.make_grid(real[:4], normalize=True)

#             writer.add_image('img_grid_fake', img_grid_fake, tb_step)
#             writer.add_image('img_grid_real', img_grid_real, tb_step)


# def clear_folders(*folders):
#     try:
#         # Clear each folder
#         for folder in folders:
#             for filename in os.listdir(folder):
#                 file_path = os.path.join(folder, filename)
#                 if os.path.isfile(file_path):
#                     os.unlink(file_path)
#                 elif os.path.isdir(file_path):
#                     shutil.rmtree(file_path)
#         print("Content of the folders cleared successfully.")
    
#     except Exception as e:
#         print(f"An error occurred: {e}")

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


