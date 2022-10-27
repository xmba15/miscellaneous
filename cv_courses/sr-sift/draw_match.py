from typing import List, Optional

import cv2
import numpy as np


def draw_match_up_down(
    query_image: np.ndarray,
    query_kpts: List[cv2.KeyPoint],
    ref_image: np.ndarray,
    ref_kpts: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    matchesMask: Optional[List[bool]] = None,
    seed: int = 2022,
):
    if len(query_image.shape) == 2 or query_image.shape[2] == 1:
        query_image = cv2.cvtColor(query_image, cv2.COLOR_GRAY2BGR)
    if len(ref_image.shape) == 2 or ref_image.shape[2] == 1:
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_GRAY2BGR)

    query_height, query_width = query_image.shape[:2]
    ref_height, ref_width = ref_image.shape[:2]
    output_shape = (query_height + ref_height, max(query_width, ref_width), 3)
    output = np.zeros(output_shape, dtype=np.uint8)
    output[:query_height, :query_width] = query_image
    output[query_height:, :ref_width] = ref_image

    def _draw_match(query_kpt, ref_kpt, color):
        DRAW_MULTIPLIER = 2
        cv2.circle(
            output,
            center=tuple([int(e) for e in query_kpt.pt]),
            radius=int(query_kpt.size / 2) * DRAW_MULTIPLIER,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            output,
            center=(int(ref_kpt.pt[0]), int(ref_kpt.pt[1] + query_height)),
            radius=int(ref_kpt.size / 2) * DRAW_MULTIPLIER,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            output,
            tuple([int(e) for e in query_kpt.pt]),
            (int(ref_kpt.pt[0]), int(ref_kpt.pt[1] + query_height)),
            color=color,
            thickness=1,
        )

    np.random.seed(2022)
    for i, cur_match in enumerate(matches):
        if matchesMask is not None and not matchesMask[i]:
            continue
        color = tuple(map(int, np.random.choice(range(256), size=3)))
        _draw_match(query_kpts[cur_match.queryIdx], ref_kpts[cur_match.trainIdx], color)

    return output


def main():
    image1_path = "data/44-l.JPG"
    image2_path = "data/44-r.JPG"
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    detector = cv2.SIFT_create()
    query_kpts, query_descs = detector.detectAndCompute(image1, None)
    ref_kpts, ref_descs = detector.detectAndCompute(image2, None)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(query_descs, ref_descs)
    matches = sorted(matches, key=lambda x: x.distance)

    output = draw_match_up_down(image1, query_kpts, image2, ref_kpts, matches[:100])
    cv2.imwrite("output.jpg", output)


if __name__ == "__main__":
    main()
