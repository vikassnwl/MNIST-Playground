import base64
import numpy as np
from PIL import Image
import io

def data_url_to_numpy_array(data_url, mode="gray"):
  """Converts a data URL to a NumPy array.

  Args:
    data_url: The data URL to convert.

  Returns:
    A NumPy array representing the image data.
  """

  # Extract the base64-encoded data
  img_str = data_url.split(',')[1]
  img_data = base64.b64decode(img_str)

  # Create a binary stream from the data
  image_file = io.BytesIO(img_data)

  # Load the image using PIL
  img = Image.open(image_file)

  # Convert the image to a NumPy array
  if mode == "gray":
    img_array = np.array(img.convert("L"))
  elif mode == "binary":
    img_array = 255-np.array(img.convert("L"))
  else:
    img_array = np.array(img)

  return img_array