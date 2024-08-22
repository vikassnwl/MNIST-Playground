import base64
import numpy as np
from PIL import Image
import io
import cv2


def data_url_to_numpy_array(data_url, mode="gray"):
    """Converts a data URL to a NumPy array.

    Args:
      data_url: The data URL to convert.

    Returns:
      A NumPy array representing the image data.
    """

    # Extract the base64-encoded data
    img_str = data_url.split(",")[1]
    img_data = base64.b64decode(img_str)

    # Create a binary stream from the data
    image_file = io.BytesIO(img_data)

    # Load the image using PIL
    img = Image.open(image_file)

    # Convert the image to a NumPy array
    if mode == "gray":
        img_array = np.array(img.convert("L"))
    elif mode == "binary":
        img_array = 255 - np.array(img.convert("L"))
    else:
        img_array = np.array(img)

    return img_array


def capture_all_cntrs(cntrs):
    min_x, min_y, max_x, max_y = 1e9, 1e9, 0, 0
    for cntr in cntrs:
        x, y, w, h = cv2.boundingRect(cntr)
        min_x = min(x, min_x)
        min_y = min(y, min_y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    return min_x, min_y, max_x - min_x, max_y - min_y


def center_digit(gray_image):
    _, gray_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    x, y, w, h = capture_all_cntrs(contours)
    if w > h:
        resize_w = 20
        resize_h = h * 20 // w
    else:
        resize_w = w * 20 // h
        resize_h = 20
    cropped_image = gray_image[y : y + h, x : x + w]
    resized_image = cv2.resize(cropped_image, (resize_w, resize_h))
    gray_h, gray_w = gray_image.shape
    resized_top = (gray_h - resize_h) // 2
    resized_bottom = resized_top + resize_h
    resized_left = (gray_w - resize_w) // 2
    resized_right = resized_left + resize_w
    canvas = np.zeros(gray_image.shape)
    canvas[resized_top:resized_bottom, resized_left:resized_right] = resized_image

    return canvas
