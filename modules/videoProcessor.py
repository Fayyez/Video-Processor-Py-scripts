from modules.imageProcessor import ImgProcessor
from cv2 import VideoCapture, imshow, waitKey
import cv2 
import os
import numpy as np
from time import time

class VideoProcessor:

    def __init__(self, base_picture: str) -> None:
        self.imgProcessor = ImgProcessor((640, 640))
        self.video = VideoCapture(0)
        self.base_frame = self.imgProcessor.load_image_in_grayscale(base_picture)
        if not self.video.isOpened():
             print("error loading the webcam")

    def __init__(self, video_path: str, base_picture: str) -> None:
        self.imgProcessor = ImgProcessor((640, 640))
        self.video_name = video_path.split('/')[-1].split('.')[0]
        self.video = VideoCapture(video_path)
        self.base_frame = self.imgProcessor.load_image_in_grayscale(base_picture)
        if not self.video.isOpened():
             print("error loading the video")

    def setVideo(self, video_path: str) -> None:
        self.video = VideoCapture(video_path)
        if not self.video.isOpened():
             print("error loading the video")

    def frames_to_data(self, threshold: float, output_dir: str):
        # get the frame from video
        # calculate the difference between the frame and the base frame
        # if the difference is greater than the threshold
        # return the frame
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        base_frame = np.array(self.base_frame)
        frame_count = 0
        timeout = True
        time_start = None
        #start the video
        while True:
            ret, frame_og = self.video.read()
            if not ret:
                print("video ended")
                break

            if timeout:
                frame = cv2.resize(frame_og, (640, 640))
                # frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                frame = np.array(frame)

                difference = self.imgProcessor.calculate_abs_diff_per_from_array(base_frame, frame)
                if difference > threshold:
                    frame_count += 1
                    cv2.imwrite(f"{output_dir}/{self.video_name}_frame_{frame_count}.jpg", frame_og)
                    timeout = False
                    time_start = time()

            if not time_start==None and time() - time_start > 10:
                timeout = True

        self.video.release()
        cv2.destroyAllWindows()
            
    def frames_to_data_with_bkg_subtraction(self, threshold: float, output_dir: str):
                    # get the frame from video
        # calculate the difference between the frame and the base frame
        # if the difference is greater than the threshold
        # return the frame
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        frame_count = 0
        timeout = True
        #start the video
        while True:
            ret, frame_og = self.video.read()
            if not ret:
                print("video ended")
                break

            if timeout:
                frame = cv2.resize(frame_og, (640, 640))

                fgbg = cv2.createBackgroundSubtractorMOG2()
                if self.imgProcessor.diff_by_subtract_bkg(frame, self.base_frame, fgbg):
                    frame_count += 1
                    cv2.imwrite(f"{output_dir}/{self.video_name}_frame_{frame_count}.jpg", frame_og)

        self.video.release()
        cv2.destroyAllWindows()

    def frames_to_data_by_ms_diff(self, output_dir: str, threshold: float):
                    # get the frame from video
        # calculate the difference between the frame and the base frame
        # if the difference is greater than the threshold
        # return the frame
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        frame_count = 0
        timeout = True
        time_start = None
        #start the video
        while True:
            ret, frame_og = self.video.read()
            if not ret:
                print("video ended")
                break
            
            if timeout:
                frame = cv2.resize(frame_og, (640, 640))

                if self.imgProcessor.get_ms_difference(frame, self.base_frame, threshold):
                    frame_count += 1
                    cv2.imwrite(f"{output_dir}/{self.video_name}_frame_{frame_count}.jpg", frame_og)
                    timeout = False
                    time_start = time()

            if not time_start == None and time() - time_start > 0.5:
                timeout = True

        self.video.release()
        cv2.destroyAllWindows()
        


# testing the code below
if __name__ == "__main__":
    processor = VideoProcessor("videos/sample.mp4")
    cap = processor.video
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("error")
        imshow("video feed", frame)
        if waitKey(10) & 0xFF == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()
