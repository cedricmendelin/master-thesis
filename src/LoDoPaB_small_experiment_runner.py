import argparse
from utils.ImageHelper import *
from utils.GATDenoiserHelper import *
import time
import os

torch.manual_seed(2022)
np.random.seed(2022)

parser = argparse.ArgumentParser()
parser.add_argument("--graph_size", type=int, default=1024)
parser.add_argument("--resolution", type=int, default=64)
parser.add_argument("--samples", type=int, default=8)

parser.add_argument("--image_path", type=str, default="src/data/limited-CT/data_png_train/")

parser.add_argument("--validation_image_path", type=str, default="src/data/limited-CT/data_png_test/")
parser.add_argument("--validation_image_count", type=int, default=100)

parser.add_argument("--use_wandb", action='store_true', default=False)
parser.add_argument("--debug_plots", action='store_true', default=False)
parser.add_argument("--wandb_project", type=str, default="LoDoPaB Small Test 2")
parser.add_argument("--save_model", action='store_true', default=False)
parser.add_argument("--model_dir", type=str, default="denoiser/small_experiments/")

parser.add_argument("--gat_layers", type=int, default=4)
parser.add_argument("--gat_heads", type=int, default=8)
parser.add_argument("--gat_dropout", type=float, default=0.05)
parser.add_argument("--gat_weight_decay", type=float, default=0.0005)
parser.add_argument("--gat_learning_rate", type=float, default=0.01)
parser.add_argument("--gat_use_conv", action='store_true', default=False)
parser.add_argument("--gat_conv_kernel", type=int, default=3)
parser.add_argument("--gat_conv_padding", type=int, default=1)
parser.add_argument("--gat_conv_N_latent", type=int, default=1)
parser.add_argument("--unet_refinement", action='store_true', default=False)
parser.add_argument("--unet_path", type=str, default="models/unet_128.pt")
parser.add_argument("--unet_train", action='store_true', default=False)

parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--gat_snr_lower", type=int, default=0)
parser.add_argument("--gat_snr_upper", type=int, default=0)
parser.add_argument("--validation_snrs", nargs="+", type=int, default=[0])
parser.add_argument("--append_validation_images", type=int, default=0)
parser.add_argument("--add_circle_padding", action='store_true', default=False)
parser.add_argument("--k_nn", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=32)

parser.add_argument('--loss', type=str, default='FBP',
                    choices=[i.name.upper() for i in Loss])

parser.add_argument("--verbose", action='store_true', default=False)

parser.add_argument("--model_state_path", type=str, default=None)
parser.add_argument("--optimizer_state_path", type=str, default=None)

parser.add_argument("--run_name", type=str, default="test")

args = parser.parse_args()


############### Start exection #################
t = time.time()

################## load images##################
image_count = args.samples
validation_image_count = args.validation_image_count

image_path = args.image_path
files = os.listdir(image_path)

RESOLUTION = args.resolution

# get images
if args.validation_image_path is None:
    x = load_images_files_rescaled(image_path, files, RESOLUTION, RESOLUTION, number=image_count +
                                   validation_image_count, num_seed=5, circle_padding=args.add_circle_padding)
    x_input = np.array(x[0:image_count])
    x_validation = np.array(
        x[image_count: image_count+validation_image_count])

else:
    validation_image_path = args.validation_image_path
    validation_files = os.listdir(validation_image_path)

    x_input = load_images_files_rescaled(image_path, files, RESOLUTION, RESOLUTION,
                                         number=image_count, num_seed=5, circle_padding=args.add_circle_padding)
    x_validation = load_images_files_rescaled(validation_image_path, validation_files, RESOLUTION,
                                              RESOLUTION, number=validation_image_count, num_seed=5, circle_padding=args.add_circle_padding)

if args.append_validation_images > 0:
    for i in range(args.append_validation_images):
        x_validation = np.append(x_validation, x_input[i])
    x_validation = x_validation.reshape(
        (validation_image_count + args.append_validation_images, RESOLUTION, RESOLUTION))

################# Load current state: #################
model_state = None
if args.model_state_path != None:
    model_state = torch.load(args.model_state_path)

optimizer_state = None
if args.optimizer_state_path != None:
    optimizer_state = torch.load(args.optimizer_state_path)

################# Initialize Denoiser: ################

denoiser = GatBase.create_fixed_images_denoiser(args, model_state, optimizer_state, args.run_name)

################# Train Denoiser: ################
model, optimizer, unet = denoiser.train(images=x_input, batch_size = args.batch_size, loss=Loss[args.loss.upper()])

################# Validate Denoiser: ################
denoiser.validate(x_validation, args.validation_snrs, 8)


################# Finish Run: ################
if args.use_wandb:
    # model_name = wandb.run.name.replace(" ", "_") + "-" + \
    #     str(args.batch_size) + "_torch_state_dict"
    denoiser.finish_wandb(time.time()-t)

if args.save_model:
    name = args.run_name

    torch.save(model.module.state_dict(), args.model_dir + name + "_gat.pt")
    torch.save(optimizer.state_dict(), args.model_dir + name + "_optimizer.pt")
    if args.unet_train:
        torch.save({
            "model_state_dict" : unet.state_dict(),
        }, args.model_dir + name + "_unet.pt")

if args.debug_plots:
    plt.show()
