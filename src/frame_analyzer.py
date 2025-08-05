import cv2

class FrameAnalyzer:
    def __init__(self, frame):
        self.frame = frame

    @staticmethod
    def is_blurry(frame, threshold=1000):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        return laplacian, variance < threshold, variance