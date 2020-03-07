import cv2 # opencv-python
import numpy as np
from skimage.filters import threshold_local # scikit-image
import imutils
import pytesseract
from PIL import Image


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right poi
    # nt will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped








# read the input image
image = cv2.imread("IMG_1105.jpg")
image = cv2.convertScaleAbs(image, alpha=1.25, beta=-100)
image = cv2.resize(image, (1000,1000))

# clone the original image
original_image = image.copy()

# resize using ratio (old height to the new height)
ratio = image.shape[0] / 500.0

image = imutils.resize(image, height=500)

#  change the color space to YUV
image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

# grap only the Y component
image_y = np.zeros(image_yuv.shape[0:2], np.uint8)
image_y[:, :] = image_yuv[:, :, 0]

# blur the image to reduce high frequency noises
image_blurred = cv2.GaussianBlur(image_y, (3, 3), 0)

# find edges in the image
edges = cv2.Canny(image_blurred, 50, 200, apertureSize=3)

# find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# draw all contours on the original image
cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
# !! Attention !! Do not draw contours on the image at this point
# I have drawn all the contours just to show below image

# to collect all the detected polygons
polygons = []

# loop over the contours
for cnt in contours:
    # find the convex hull
    hull = cv2.convexHull(cnt)

    # compute the approx polygon and put it into polygons
    polygons.append(cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), False))


# sort polygons in desc order of contour area
sortedPoly = sorted(polygons, key=cv2.contourArea, reverse=True)

# draw points of the intersection of only the largest polyogon with red color
cv2.drawContours(image, sortedPoly[0], -1, (0, 0, 255), 5)

# get the contours of the largest polygon in the image
simplified_cnt = sortedPoly[0]

# check if the polygon has four point
# if len(simplified_cnt) == 4:
    # trasform the prospective of original image

x=len(simplified_cnt)
cropped_image = four_point_transform(original_image, simplified_cnt.reshape(x, 2) * ratio)
#cropped_image.save(cropped_image.jpg)
cv2.imwrite('cropped_image.jpg', cropped_image) 

# Binarize the cropped image
gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
T = threshold_local(gray_image, 21, offset=10, method="gaussian")
binarized_image = (gray_image > T).astype("uint8") * 255
filename = 'savedImage.jpg'
cv2.imwrite(filename, binarized_image)
'''text = pytesseract.image_to_string(Image.open(cropped_image))
os.remove(filename)
print(text)'''



try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """
    text = pytesseract.image_to_string(Image.open(filename))  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    return text
print(ocr_core('IMG_1105.jpg'))

# Show images
cv2.imshow("Original", original_image)
cv2.imshow("Scanned", binarized_image)
cv2.imshow("Cropped", cropped_image)
cv2.waitKey(0)

