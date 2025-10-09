def input_code(input_type):
    if input_type == 'ImageData':
        ret = """\
import cv2
import numpy as np

# Path to the image
image_path = "D:\\Bilder\\m_ali.jpg"

# Read the image
input_img = cv2.imread(image_path)
"""
        return ret
    elif input_type == 'TextData':
        ret="""\
text = "Replace this with your text here"
        """
        return ret
    else:

        return ''
