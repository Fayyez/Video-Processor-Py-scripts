import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

class ImgProcessor:

    def __init__(self, size: tuple = (640, 640)):
        self.size = size

    def load_image_in_grayscale(self, file_path: str):
        # load image in grayscale and resize to the given size
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Could not open or find the image {file_path}")
        img_resized = cv2.resize(img, self.size)
        return img_resized
    
    def calculate_abs_diff_per_from_array(self, img1_array: np.array, img2_array: np.array):
        diff = np.abs(img1_array - img2_array)
        percentage_diff = np.sum(diff) / np.sum(img1_array) * 100
        return percentage_diff
    
    def calculate_abs_diff_per_from_path(self, img1_path: str, img2_path: str):
        # load the images
        img1 = self.load_image_in_grayscale(img1_path)
        img2 = self.load_image_in_grayscale(img2_path)
        # return the percentage difference between two images
        img1_array = np.array(img1)
        img2_array = np.array(img2)

        return self.calculate_abs_diff_per_from_array(img1_array, img2_array)
    
    def diff_by_subtract_bkg(self, frame, bkg_frame, fgbg):
        fgmask = fgbg.apply(bkg_frame)
        fgmask = fgbg.apply(frame)
        diff = np.count_nonzero(fgmask)
        return diff > 500000 or diff < 200000

    def diff_by_ssim(self, frame, bkg_frame):
        score, diff = ssim(bkg_frame, frame, full=True)
        diff = (diff * 255).astype("uint8")
        # return root mean square difference
        return np.mean(diff)
    
    def get_ms_difference(self, frame, bkg_frame, threshold):
        frame = frame / 255.0
        bkg_frame = bkg_frame / 255.0
        return np.mean((frame - bkg_frame) ** 2) * 1000 > threshold



# Example usage:
if __name__ == "__main__":
    minimal = 'D:/dataset generation/videos/empty2.jpg'
    empty = 'D:/dataset generation/videos/emptyBkg.jpg'
    template = 'D:/dataset generation/videos/template1.jpg'
    car = 'D:/dataset generation/videos/car.jpg'
    cars = 'D:/dataset generation/videos/cars.jpg'
    maximum = 'D:/dataset generation/videos/maxVehicle.jpg'

    processor = ImgProcessor()

    bkg_frame = processor.load_image_in_grayscale(empty)
    minimal = processor.load_image_in_grayscale(minimal)
    template = processor.load_image_in_grayscale(template)
    car = processor.load_image_in_grayscale(car)
    cars = processor.load_image_in_grayscale(cars)
    maximum = processor.load_image_in_grayscale(maximum)

    fgbf = cv2.createBackgroundSubtractorMOG2()
    
    print('SUMMARY:\n')
    ## using structural difference
    # print('minimal', processor.diff_by_ssim(minimal, bkg_frame))
    # print('template', processor.diff_by_ssim(template, bkg_frame))
    # print('car', processor.diff_by_ssim(car, bkg_frame))
    # print('cars', processor.diff_by_ssim(cars, bkg_frame))
    # print('maximum', processor.diff_by_ssim(maximum, bkg_frame))
    print('\n')
    # using background subtraction
    # print('minimal', processor.diff_by_subtract_bkg(minimal, bkg_frame, fgbf))
    # print('template', processor.diff_by_subtract_bkg(template, bkg_frame, fgbf))
    # print('car', processor.diff_by_subtract_bkg(car, bkg_frame, fgbf))
    # print('cars', processor.diff_by_subtract_bkg(cars, bkg_frame, fgbf))
    # print('maximum', processor.diff_by_subtract_bkg(maximum, bkg_frame, fgbf))
    # print('\n')
    ## usng mean square difference
    print('minimal', processor.get_ms_difference(minimal, bkg_frame))
    print('template', processor.get_ms_difference(template, bkg_frame))
    print('car', processor.get_ms_difference(car, bkg_frame))
    print('cars', processor.get_ms_difference(cars, bkg_frame))
    print('maximum', processor.get_ms_difference(maximum, bkg_frame))
