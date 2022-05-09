import argparse
from utils.ImageHelper import *
from utils.GATDenoiserHelper import *
import time

# Prepare arguments
parser = argparse.ArgumentParser()
parser.add_argument("--graph_size", type=int, default=1024)
parser.add_argument("--samples", type=int, default=1024)
parser.add_argument("--resolution", type=int, default=64)

parser.add_argument("--validation_image_path", type=str, default="src/toyimages_validation/")
parser.add_argument("--validation_image_count", type=int, default=18)
parser.add_argument("--use_wandb", type=bool, default=False)
parser.add_argument("--wandb_project", type=str, default="Some Test")
parser.add_argument("--save_model", type=bool, default=True)
parser.add_argument("--model_dir", type=str, default="denoiser/")
parser.add_argument("--gat_layers", type=int, default=4)
parser.add_argument("--gat_heads", type=int, default=16)
parser.add_argument("--gat_dropout", type=float, default=0.05)
parser.add_argument("--gat_weight_decay", type=float, default=0.0005)
parser.add_argument("--gat_learning_rate", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--gat_snr_lower", type=int, default=10)
parser.add_argument("--gat_snr_upper", type=int, default=10)
parser.add_argument("--validation_snrs", nargs="+", type=int, default=[10, 25])
parser.add_argument("--k_nn", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--verbose", type=bool, default=True)

parser.add_argument('--loss', type=str, default='SINO',
                    choices=[i.name.upper() for i in Loss])


args = parser.parse_args()


############### Start exection #################
t = time.time()

################## load images##################
RESOLUTION = args.resolution

# get images

validation_image_path = args.validation_image_path
validation_files = os.listdir(validation_image_path)

validation_image_count = args.validation_image_count
x_validation = load_images_files_rescaled(validation_image_path, validation_files, RESOLUTION, RESOLUTION, number=validation_image_count, num_seed=5, circle_padding=False)

################# Initialize Denoiser: ################

denoiser = GatBase.create_toyimage_dynamic_denoiser(args)

################# Train Denoiser: ################
model = denoiser.train(batch_size=args.batch_size, loss=Loss[args.loss.upper()])

################# Validate Denoiser: ################
denoiser.validate(x_validation, args.validation_snrs, args.batch_size)

################# Finish Run: ################
if args.use_wandb:
    model_name = wandb.run.name.replace(" ", "_") + "-" + \
        str(args.batch_size) + "_torch_state_dict"
    denoiser.finish_wandb(time.time()-t)

if args.save_model:
    if not args.use_wandb:
        model_name = "out_torch_state_dict_model"

    if args.model_dir is None:
        torch.save(model.module.state_dict(), model_name)
    else:
        torch.save(model.module.state_dict(), args.model_dir + model_name)

