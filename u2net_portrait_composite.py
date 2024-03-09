# Standard library imports
import argparse
import glob
import os
from pathlib import Path

# Third party imports
import numpy as np
from PIL import Image
from skimage import io, transform
from skimage.filters import gaussian
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

# Local imports
from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import U2NET  # full size version 173.6 MB


SRC_DIR = Path("./test_data/test_portrait_images/your_portrait_im")
MODEL_DIR = Path("./saved_models/u2net_portrait/u2net_portrait.pth")
DST_DIR = Path("./test_data/test_portrait_images/your_portrait_results")

# normalize the predicted SOD probability map
def norm_pred(d: torch.Tensor) -> torch.Tensor:
    """
    Normalize the input tensor.
    Arguments:
    - d: The input tensor to be normalized.
    Returns:
    - dn: The normalized tensor.
    """
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir, sigma=2, alpha=0.5):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    image = io.imread(image_name)
    pd = transform.resize(predict_np, image.shape[0:2], order=2)
    pd = pd / (np.amax(pd) + 1e-8) * 255
    pd = pd[:, :, np.newaxis]

    print(image.shape)
    print(pd.shape)

    ## fuse the orignal portrait image and the portraits into one composite image
    ## 1. use gaussian filter to blur the orginal image
    sigma = sigma
    image = gaussian(image, sigma=sigma, preserve_range=True)

    ## 2. fuse these orignal image and the portrait with certain weight: alpha
    alpha = alpha
    im_comp = image * alpha + pd * (1 - alpha)
    im = Image.fromarray(np.round(im_comp).astype(np.uint8))
    print(im_comp.shape)
    img_name = image_name.split(os.sep)[-1]
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]
    dst = (
        d_dir
        + "/"
        + imidx
        + "_sigma_"
        + str(sigma)
        + "_alpha_"
        + str(alpha)
        + "_composite.png"
    )
    im.save(dst)


def fuse_input_and_portrait(
    input_image_src: Path,
    portrait: torch.Tensor,
    sigma: float = 2,
    alpha: float = 0.5,
) -> Image:
    """
    Fuse the original image and the portrait image into one composite image.

    Arguments:
        image: The path to the original image.
        portrait: The portrait image.
        sigma: The standard deviation of the Gaussian filter applied to the original image.
        alpha: The weight of the original image in the composite image.
    Returns:
        The composite image.
    """
    # Prepare original image for fusion.
    image = io.imread(input_image_src)
    image = gaussian(image, sigma=sigma, preserve_range=True)
    # Prepare portrait for fusion.
    portrait = portrait.squeeze().cpu().data.numpy()
    portrait = transform.resize(portrait, image.shape[0:2], order=2)
    portrait = portrait / (np.amax(portrait) + 1e-8) * 255
    portrait = portrait[:, :, np.newaxis]
    # Fuse original image and portrait using weight alpha
    composite_image = image * alpha + portrait * (1 - alpha)
    return Image.fromarray(np.round(composite_image).astype(np.uint8))


def main():

    parser = argparse.ArgumentParser(description="image and portrait composite")
    parser.add_argument("--src_dir", "-i", type=Path, default=SRC_DIR, help="Path to the input image directory.")
    parser.add_argument("--model_dir", "-j", type=Path, default=MODEL_DIR, help="Path to the model directory.")
    parser.add_argument("--dst_dir", "-o", type=Path, default=DST_DIR, help="Path to the output image directory.")
    parser.add_argument("--sigma", "-s", type=float, default=2.0, help="Sigma value for gaussian filter.")
    parser.add_argument("--alpha", "-a", type=float, default=0.5, help="Alpha value for image and portrait fusion.")
    parser.add_argument("--save_all", "-sa", action="store_true", help="Save all the results, not just best ones.")
    args = parser.parse_args()
    
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    # Fetch input image paths and make output directory
    srcs = list(args.src_dir.iterdir())
    args.dst_dir.mkdir(exist_ok=True, parents=True)
    print(f"Number of images: {len(srcs)}")

    # Instantiate dataset and dataloader
    test_salobj_dataset = SalObjDataset(
        img_name_list=srcs,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(512), ToTensorLab(flag=0)]),
    )
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1
    )

    # Load model and set to evaluation
    print("...load U2NET---173.6 MB")
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(args.model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # Run inference for each image
    for i_test, data_test in enumerate(test_salobj_dataloader):
        inputs_test = data_test["image"]
        inputs_test = inputs_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        # Infer        
        preds = net(inputs_test)
        # Iterate over each prediction, fuse input with prediction, and save composite output
        for ix, pred in enumerate(preds, start=1):
            pred = 1.0 - pred[:, 0, :, :]
            pred = norm_pred(pred)
            
            # fuse the original image and the portrait
            composite_image = fuse_input_and_portrait(
                srcs[i_test],
                pred,
                sigma=float(args.sigma),
                alpha=float(args.alpha),
            )
            # save results to test_results folder
            fn = f"{srcs[i_test].stem}_d{ix}_a{args.alpha}_s{args.sigma}_composite.png"
            dst = args.dst_dir / fn
            print(f"Saving composite image to {dst}.")
            composite_image.save(str(dst))
            if not args.save_all:
                break


if __name__ == "__main__":
    main()
