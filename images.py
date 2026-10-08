from dataclasses import dataclass

import cv2
import numpy


@dataclass
class ImageLayer:
    image: numpy.ndarray
    scale: float

def get_images_with_scales(image, scales):
    images = [ImageLayer(image, 1)]
    for scale in scales:
        layer = cv2.resize(images[-1].image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        images.append(ImageLayer(layer, scale))

    for image in images:
        cv2.imshow('image', image)
        print(image.image.shape)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    return images

# get_images_with_scales(cv2.imread('image.jpeg', cv2.IMREAD_GRAYSCALE), scales)