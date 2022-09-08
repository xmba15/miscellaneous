import cv2

lcolor = cv2.imread("data/44-l.JPG")
rcolor = cv2.imread("data/44-r.JPG")

left = cv2.imread("data/44-l.JPG", 0)
right = cv2.imread("data/44-r.JPG", 0)

lsd = cv2.line_descriptor.LSDDetector_createLSDDetector()
bd = cv2.line_descriptor.BinaryDescriptor_createBinaryDescriptor()
matcher = cv2.line_descriptor.BinaryDescriptorMatcher()

lkeylines = lsd.detect(left, 2, 1)
lkeylines, ldescriptors = bd.compute(left, lkeylines)

rkeylines = lsd.detect(right, 2, 1)
rkeylines, rdescriptors = bd.compute(right, rkeylines)

matches = matcher.match(ldescriptors, rdescriptors)
matches = sorted(matches, key=lambda match: match.distance)


def keyLinesToKeyPoints(keyLines):
    keyPoints = []
    for keyLine in keyLines:
        curKeyPoint = cv2.KeyPoint()
        curKeyPoint.pt = keyLine.pt
        keyPoints.append(curKeyPoint)

    return keyPoints


tmp = cv2.drawMatches(
    lcolor,
    keyLinesToKeyPoints(lkeylines),
    rcolor,
    keyLinesToKeyPoints(rkeylines),
    matches[:10],
    None,
    flags=2,
)

for _match in matches:
    lkeyline = lkeylines[_match.queryIdx]

cv2.line_descriptor.drawLineMatches(
    left, lkeylines, right, rkeylines, matches, "matchesmask", matches
)

# print(tmp)

# loutput = cv2.line_descriptor.drawKeylines(lcolor, lkeylines)
# cv2.imshow("left keylines", loutput)
# cv2.waitKey(0)


cv2.imshow("matched", tmp)
cv2.waitKey(0)
cv2.destroyAllWindows()
