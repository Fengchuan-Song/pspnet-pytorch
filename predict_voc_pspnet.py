import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from nets.pspnet import PSPNet
from utils.utils import cvtColor, preprocess_input, resize_image


IMAGE_EXTENSIONS = (".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff")
VOC_COLORS = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Predict PSPNet semantic segmentation masks in VOC format.")
    parser.add_argument("--model_path", type=str, default="/data/PSPNet/PSPNet/weights/best_epoch_weights_ep034_mIoU0.767.pth", help="Path to trained .pth weights.")
    parser.add_argument("--input_path", type=str, default="/data_ssd/datasets/WaterScenes/MIPC_SemanticSegmentation/2007_test.txt", help="RGB image file, image directory, or txt file.")
    parser.add_argument("--dataset_path", type=str, default="/data_ssd/datasets/WaterScenes", help="Dataset root used when input txt only contains image names.")
    parser.add_argument("--image_dir", type=str, default="images", help="Image directory under dataset_path.")
    parser.add_argument("--output_dir", type=str, default="/data/PSPNet/predict_results/SegmentationClass", help="VOC-style output directory.")
    parser.add_argument("--num_classes", type=int, default=9, help="Number of classes, including background.")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["mobilenet", "resnet50"], help="PSPNet backbone.")
    parser.add_argument("--downsample_factor", type=int, default=8, choices=[8, 16], help="PSPNet downsample factor used during training.")
    parser.add_argument("--input_shape", type=int, nargs=2, default=[320, 320], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--aux_branch", action="store_true", help="Build PSPNet with the auxiliary branch. Use this only if training used aux_branch=True.")
    parser.add_argument("--multi_task", action="store_true", help="Build PSPNet with object and shoreline heads.")
    parser.add_argument("--object_num_classes", type=int, default=None, help="Object head classes when --multi_task is used.")
    parser.add_argument("--shoreline_num_classes", type=int, default=None, help="Shoreline head classes when --multi_task is used.")
    parser.add_argument("--task", type=str, default="object", choices=["object", "shoreline"], help="Which multi-task head to save when --multi_task is used.")
    parser.add_argument("--strict", action="store_true", help="Strictly require all weight keys to match.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if it is available.")
    parser.add_argument("--copy_images", action="store_true", help="Copy source images to output JPEGImages.")
    return parser.parse_args()


def resolve_txt_image_path(line, txt_path, dataset_path, image_dir):
    raw_path = Path(line)
    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(txt_path.parent / raw_path)
        if dataset_path:
            candidates.append(Path(dataset_path) / image_dir / raw_path)
            candidates.append(Path(dataset_path) / raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if raw_path.suffix.lower() not in IMAGE_EXTENSIONS:
        for suffix in IMAGE_EXTENSIONS:
            for candidate in candidates:
                candidate_with_suffix = candidate.with_suffix(suffix)
                if candidate_with_suffix.exists():
                    return candidate_with_suffix

    return candidates[0]


def collect_image_paths(input_path, dataset_path=None, image_dir="images"):
    path = Path(input_path)
    if path.is_file() and path.suffix.lower() == ".txt":
        image_paths = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                image_paths.append(resolve_txt_image_path(line, path, dataset_path, image_dir))
        return image_paths
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def clean_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[7:]
        state_dict[key] = value
    return state_dict


def load_model(args, device):
    model = PSPNet(
        num_classes=args.num_classes,
        downsample_factor=args.downsample_factor,
        backbone=args.backbone,
        pretrained=False,
        aux_branch=args.aux_branch,
        multi_task=args.multi_task,
        object_num_classes=args.object_num_classes,
        shoreline_num_classes=args.shoreline_num_classes,
    )

    checkpoint = torch.load(args.model_path, map_location=device)
    incompatible = model.load_state_dict(clean_state_dict(checkpoint), strict=args.strict)
    if not args.strict:
        missing_keys, unexpected_keys = incompatible
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")

    model.eval()
    model.to(device)
    return model


def select_output(output, multi_task=False, task="object"):
    if not isinstance(output, (tuple, list)):
        return output

    if multi_task:
        return output[0] if task == "object" else output[1]

    return output[-1]


def predict_mask(model, image, input_shape, device, multi_task=False, task="object"):
    image = cvtColor(image)
    original_h, original_w = np.array(image).shape[:2]

    image_data, resized_w, resized_h = resize_image(image, (input_shape[1], input_shape[0]))
    image_data = np.expand_dims(
        np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),
        0,
    )

    with torch.no_grad():
        images = torch.from_numpy(image_data).to(device)
        pred = select_output(model(images), multi_task=multi_task, task=task)[0]
        pred = F.softmax(pred.permute(1, 2, 0), dim=-1).cpu().numpy()
        top = int((input_shape[0] - resized_h) // 2)
        left = int((input_shape[1] - resized_w) // 2)
        pred = pred[top: top + resized_h, left: left + resized_w]
        pred = cv2.resize(pred, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        pred = pred.argmax(axis=-1).astype(np.uint8)
    return pred


def save_voc_png(mask, save_path):
    palette = []
    for color in VOC_COLORS:
        palette.extend(color)
    palette.extend([0] * (256 * 3 - len(palette)))

    image = Image.fromarray(mask, mode="P")
    image.putpalette(palette)
    image.save(save_path)


def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    image_paths = collect_image_paths(args.input_path, args.dataset_path, args.image_dir)
    if len(image_paths) == 0:
        raise ValueError(f"No image files found in {args.input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir.parent / "JPEGImages"
    if args.copy_images:
        img_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args, device)

    for image_path in tqdm(image_paths, desc="Predict"):
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_id = image_path.stem
        image = Image.open(image_path).convert("RGB")
        mask = predict_mask(model, image, args.input_shape, device, args.multi_task, args.task)

        save_voc_png(mask, output_dir / f"{image_id}.png")
        if args.copy_images:
            shutil.copy2(image_path, img_dir / image_path.name)

    print(f"Saved VOC prediction masks to: {output_dir}")


if __name__ == "__main__":
    main()
