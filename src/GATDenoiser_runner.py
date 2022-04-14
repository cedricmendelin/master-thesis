import argparse
from utils.CoCoDataset import *
from utils.GATDenoiser import *
import time


parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=1024)
parser.add_argument("--resolution", type=int, default=128)
parser.add_argument("--input_image_count", type=int, default=8)
parser.add_argument("--image_path", type=str, default="src/toyimages/128/")
parser.add_argument("--validation_image_path", type=str, default=None)
parser.add_argument("--validation_image_count", type=int, default=5)
parser.add_argument("--use_wandb", type=bool, default=False)
parser.add_argument("--debug_plots", type=bool, default=False)
parser.add_argument("--wandb_project", type=str)
parser.add_argument("--save_model", type=bool, default=True)
parser.add_argument("--gat_layers", type=int, default=3)
parser.add_argument("--gat_heads", type=int, default=1)
parser.add_argument("--gat_dropout", type=float, default=0.05)
parser.add_argument("--gat_weight_decay", type=float, default=0.0005)
parser.add_argument("--gat_learning_rate", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--gat_snr_lower", type=int, default=-5)
parser.add_argument("--gat_snr_upper", type=int, default=20)
parser.add_argument("--validation_snrs", nargs="+",
                    type=int, default=[-5, 2, 10, 25])
parser.add_argument("--append_validation_images", type=int, default=0)
parser.add_argument("--add_circle_padding", type=bool, default=True)
parser.add_argument("--k_nn", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=256)

parser.add_argument("--verbose", type=bool, default=False)

args = parser.parse_args()


t = time.time()

image_count = args.input_image_count
validation_image_count = args.validation_image_count

image_path = args.image_path
files = os.listdir(image_path)

N = args.samples
RESOLUTION = args.resolution

# get images
if args.validation_image_path is None:
    x = load_images_files(image_path, files, RESOLUTION, RESOLUTION, number=image_count +
                          validation_image_count, num_seed=5, circle_padding=args.add_circle_padding)
    x_input = np.array(x[0:image_count])
    x_validation = np.array(
        x[image_count-1: image_count+validation_image_count])

else:
    validation_image_path = args.validation_image_path
    validation_files = os.listdir(validation_image_path)

    x_input = load_images_files(image_path, files, RESOLUTION, RESOLUTION,
                                     number=image_count, num_seed=5, circle_padding=args.add_circle_padding)
    x_validation = load_images_files(validation_image_path, validation_files, RESOLUTION,
                                          RESOLUTION, number=validation_image_count, num_seed=5, circle_padding=args.add_circle_padding)

if args.append_validation_images > 0:
    for i in range(args.append_validation_images):
        x_validation = np.append(x_validation, x_input[i])
    x_validation = x_validation.reshape(
        (validation_image_count + args.append_validation_images, RESOLUTION, RESOLUTION))

denoiser = GatDenoiser(
  x_input, 
  N,
  RESOLUTION,
  args.k_nn,
  args.epochs,
  layers=args.gat_layers, 
  heads=args.gat_heads, 
  dropout=args.gat_dropout,
  weight_decay=args.gat_weight_decay,
  learning_rate=args.gat_learning_rate,
  snr_lower=args.gat_snr_lower, 
  snr_upper=args.gat_snr_upper, 
  debug_plot = args.debug_plots, 
  use_wandb = args.use_wandb,
  verbose=args.verbose,
  wandb_project=args.wandb_project)


model = denoiser.train(args.batch_size)
denoiser.validate(x_validation, args.validation_snrs)

if args.use_wandb:
  denoiser.finish_wandb(time.time()-t)

if args.save_model:
    torch.save(model.module.state_dict(), "out_torch_state_dict_model")

if args.debug_plots:
    plt.show()

