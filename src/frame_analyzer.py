import cv2
import numpy as np
from data import colors

class FrameAnalyzer:
    def __init__(self, frame):
        self.frame = frame

    @staticmethod
    def fft_sharpness(gray_cell, low_freq_size=10):
        f = np.fft.fft2(gray_cell)
        fshift = np.fft.fftshift(f)
        mag_spectrum = np.abs(fshift)

        h, w = mag_spectrum.shape
        cX, cY = (int(w // 2), int(h // 2))
        mag_spectrum[cY - low_freq_size: cY + low_freq_size, cX - low_freq_size: cX + low_freq_size] = 0

        return np.mean(mag_spectrum)

    @staticmethod
    def tenengrad(gray_cell):
        sx = cv2.Sobel(gray_cell, cv2.CV_64F, 1, 0, ksize=5)
        sy = cv2.Sobel(gray_cell, cv2.CV_64F, 0, 1, ksize=5)
        return cv2.magnitude(sx, sy).mean()


    @staticmethod
    def is_blurry(frame, lap_threshold, fft_threshold, ten_threshold, blur_percentage, grid_size, low_contrast_threshold=15,
                   dark_threshold=10):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (np.mean(gray) < dark_threshold) or (np.std(gray) < low_contrast_threshold):
            return False, []

        height, width = gray.shape
        num_rows, num_cols = grid_size
        cell_height = height // num_rows
        cell_width = width // num_cols
        num_blurry = 0
        num_low_info = 0
        total_cells = num_rows * num_cols
        blur_map = []

        for i in range(num_rows):
            for j in range(num_cols):
                y_start, y_end = i * cell_height, (i + 1) * cell_height # (num_of_rows) * (height of the cell)
                x_start, x_end = j * cell_width, (j + 1) * cell_width # (num_of_cols) * (width of the cell)
                cell = gray[y_start: y_end, x_start : x_end] # slice the original frame into those cells
                if cell.size == 0:
                    continue

                if (np.mean(cell) < dark_threshold) or (np.std(cell) < low_contrast_threshold):
                    color = colors.COLOR_GRAY
                    rect = (x_start, y_start, x_end, y_end, color)
                    blur_map.append((rect, 0, 0, 0))
                    num_low_info += 1
                    continue

                laplacian = cv2.Laplacian(cell, cv2.CV_64F)
                variance = laplacian.var()
                fft_score = FrameAnalyzer.fft_sharpness(cell)
                ten_score = FrameAnalyzer.tenengrad(cell)

                color = (0, 255, 0) # color cell green if not blurry
                if (variance < lap_threshold) and (fft_score < fft_threshold) and (ten_score < ten_threshold): # mark blurry if both laplacian and fourier transform values are low
                    color = (0, 0, 255) # otherwise its red
                    num_blurry += 1
                rect = (x_start, y_start, x_end, y_end, color)
                blur_map.append((rect, variance, fft_score, ten_score))

        detectable_cells = total_cells - num_low_info
        if detectable_cells == 0:
            return False, blur_map

        final_blur_percentage = (num_blurry / detectable_cells) * 100

        return final_blur_percentage > blur_percentage, blur_map