import time

import cv2
import faiss
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from kornia_moons.feature import *


def extract_features(
    img,
    detector=cv2.SIFT_create(),
    affine=KF.LAFAffNetShapeEstimator(True).eval(),
    descriptor=KF.HardNet8(True).eval(),
):
    kpts = detector.detect(img, None)
    with torch.no_grad():
        timg = K.image_to_tensor(img, False).float() / 255.0
        lafs = laf_from_opencv_SIFT_kpts(kpts)
        patches = KF.extract_patches_from_pyramid(timg, lafs, 32)
        B, N, CH, H, W = patches.size()
        descs = (
            descriptor(patches.view(B * N, CH, H, W))
            .view(B * N, -1)
            .detach()
            .cpu()
            .numpy()
        )
    return kpts, descs

lcolor = cv2.imread("data/44-l.JPG")
rcolor = cv2.imread("data/44-r.JPG")

left = cv2.imread("data/44-l.JPG", 0)
right = cv2.imread("data/44-r.JPG", 0)

kpt_detector = cv2.SIFT_create()
lkpts, ldescs = kpt_detector.detectAndCompute(left, None)
# lkpts: Tuple[cv2.KeyPoint]
# ldescs: np.ndarray of shape (num_kpts, descriptor_dimension), float32

kl_kpts, kl_descs = extract_features(left)
# kl_kpts float

kr_kpts, kr_descs = extract_features(right)
kr_descs = np.vstack((kr_descs, kr_descs))

print("check point")
tic = time.perf_counter()
dimension = kr_descs.shape[1]
index1 = faiss.IndexFlatL2(dimension)
index1.add(kl_descs)

index2 = faiss.IndexFlatL2(dimension)
index2.add(kr_descs)

D1, I1 = index1.search(kr_descs, 1)
D2, I2 = index2.search(kl_descs, 1)

matches = []
for idx, value in enumerate(I2):
    if I1[value[0]] == idx:
        match = cv2.DMatch()
        match.imgIdx = 0
        match.queryIdx = idx
        match.trainIdx = value[0]
        matches.append(match)
toc = time.perf_counter()
print(toc - tic)
print(len(matches))

matches1_img = cv2.drawMatches(lcolor, kl_kpts, rcolor, kr_kpts, matches[:100], None)
cv2.imwrite("matches1_img.jpg", matches1_img)

tic = time.perf_counter()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches2 = bf.match(kl_descs, kr_descs)
print(len(matches2))
toc = time.perf_counter()
print(toc - tic)

matches2_img = cv2.drawMatches(lcolor, kl_kpts, rcolor, kr_kpts, matches2[:100], None)
cv2.imwrite("matches2_img.jpg", matches2_img)
