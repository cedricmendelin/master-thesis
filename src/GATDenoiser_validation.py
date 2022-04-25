import argparse
from utils.CoCoDataset import *
from utils.GATDenoiser import *
import time


parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=1024)
parser.add_argument("--resolution", type=int, default=192)

parser.add_argument("--validation_image_path", type=str, default=None)
parser.add_argument("--validation_image_count", type=int, default=5)
parser.add_argument("--use_wandb", type=bool, default=False)
parser.add_argument("--debug_plots", type=bool, default=False)
parser.add_argument("--wandb_project", type=str)


parser.add_argument("--gat_layers", type=int, default=3)
parser.add_argument("--gat_heads", type=int, default=1)
parser.add_argument("--gat_dropout", type=float, default=0.05)


parser.add_argument("--validation_snrs", nargs="+", type=int, default=[10, 25])

parser.add_argument("--add_circle_padding", type=bool, default=True)
parser.add_argument("--k_nn", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=256)

parser.add_argument("--verbose", type=bool, default=True)

parser.add_argument("--model_state_path", type=str)

args = parser.parse_args()



if args.model_path is None:
  raise RuntimeError("No model available to validate")




N = args.samples
RESOLUTION = args.resolution

# get images
validation_image_path = args.validation_image_path
validation_image_count = args.validation_image_count
validation_files = os.listdir(validation_image_path)

x_validation = load_images_files_rescaled(validation_image_path, validation_files, RESOLUTION,
                                          RESOLUTION, number=validation_image_count, num_seed=5, circle_padding=args.add_circle_padding)


model_state = torch.load(args.model_state_path)

validator = GatDenoiser.create_validator(
    args,
    model_state)


validator.validate(x_validation, args.validation_snrs, args.batch_size)

if args.use_wandb:
    model_name = wandb.run.name + "-" + \
        str(args.batch_size) + "_torch_state_dict"
    denoiser.finish_wandb(time.time()-t)

if args.save_model:
    if not args.use_wandb:
        model_name = "out_torch_state_dict_model"

    if args.model_dir is None:
        torch.save(model.module.state_dict(), model_name)
    else:
        torch.save(model.module.state_dict(), args.model_dir + model_name)

if args.debug_plots:
    plt.show()
