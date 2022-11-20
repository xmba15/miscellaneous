import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch


def main():
    image_l_paths = ["data/44-l.JPG", "data/46-l.jpg"]
    image_r_paths = ["data/44-r.JPG", "data/46-r.jpg"]

    l_images = [cv2.imread(image_path, 0) for image_path in image_l_paths]
    r_images = [cv2.imread(image_path, 0) for image_path in image_r_paths]

    l_images = [cv2.resize(image, (640, 480)) for image in l_images]
    r_images = [cv2.resize(image, (640, 480)) for image in r_images]

    concat_l_images = np.concatenate(
        [image[None, None, :, :] for image in l_images], axis=0
    )
    concat_l_images = torch.from_numpy(concat_l_images).float() / 255

    concat_r_images = np.concatenate(
        [image[None, None, :, :] for image in r_images], axis=0
    )
    concat_r_images = torch.from_numpy(concat_r_images).float() / 255

    matcher = KF.LoFTR(pretrained="outdoor").eval()

    with torch.no_grad():
        corr = matcher({"image0": concat_l_images, "image1": concat_r_images})

        print(corr.keys())
        # dict_keys(['keypoints0', 'keypoints1', 'confidence', 'batch_indexes'])

        print(corr["keypoints0"][corr["batch_indexes"] == 0].shape)


if __name__ == "__main__":
    main()
