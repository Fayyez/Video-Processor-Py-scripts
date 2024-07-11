from modules.videoProcessor import VideoProcessor

# initialize processor here
# processor = ImgProcessor((640,640))
# print(processor.size)
# print(processor.calculate_diff_per_from_path(img1_path="videos/emptyBkg.jpg", img2_path='videos/template1.jpg'))

processor = VideoProcessor('videos/fastforwarded.mp4', 'videos/emptyBkg.jpg')
processor.frames_to_data_by_ms_diff('output', 15)
