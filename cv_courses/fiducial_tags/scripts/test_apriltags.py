#!/usr/bin/env python
import dt_apriltags as at
import cv2


def get_args():
    import argparse

    parser = argparse.ArgumentParser("")
    parser.add_argument("--image_path", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()
    img = cv2.imread(args.image_path)
    assert img is not None, "failed to load {}".format(args.image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    at_detector = at.Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )
    tags = at_detector.detect(gray_img)
    for tag in tags:
        (ptA, ptB, ptC, ptD) = [(int(e1), int(e2)) for (e1, e2) in tag.corners]
        cv2.line(img, ptA, ptB, (0, 255, 0), 2)
        cv2.line(img, ptB, ptC, (0, 255, 0), 2)
        cv2.line(img, ptC, ptD, (0, 255, 0), 2)
        cv2.line(img, ptD, ptA, (0, 255, 0), 2)
        (c_x, c_y) = (int(tag.center[0]), int(tag.center[1]))
        cv2.circle(img, (c_x, c_y), 5, (0, 0, 255), -1)

        tagFamily = tag.tag_family.decode("utf-8")
        cv2.putText(
            img,
            tagFamily,
            (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
