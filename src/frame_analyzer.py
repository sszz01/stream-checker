import cv2
import numpy as np

##################################### FREEZE DETECTION #####################################

# def is_frozen(frame1, frame2, freeze_threshold):
#     frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#     frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#
#     diff = cv2.absdiff(frame1_gray, frame2_gray)
#     freeze_score = np.sum(diff > freeze_threshold)
#
#     return freeze_score > 0


##################################### BLUR DETECTION #####################################


def fft_sharpness(gray_cell, low_freq_size=10):
    f = np.fft.fft2(gray_cell)
    fshift = np.fft.fftshift(f)
    mag_spectrum = np.abs(fshift)

    h, w = mag_spectrum.shape
    cX, cY = (int(w // 2), int(h // 2))
    mag_spectrum[cY - low_freq_size: cY + low_freq_size, cX - low_freq_size: cX + low_freq_size] = 0

    return np.mean(mag_spectrum)


def tenengrad(gray_cell):
    sx = cv2.Sobel(gray_cell, cv2.CV_64F, 1, 0, ksize=5)
    sy = cv2.Sobel(gray_cell, cv2.CV_64F, 0, 1, ksize=5)
    return cv2.magnitude(sx, sy).mean()


def is_blurry(frame1, frame2, rel_threshold, blur_percentage, grid_size):
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    num_rows, num_cols = grid_size
    cell_height = height // num_rows
    cell_width = width // num_cols
    num_blurry = 0
    total_cells = num_rows * num_cols
    blur_map = []

    for i in range(num_rows):
        for j in range(num_cols):
            y_start, y_end = i * cell_height, (i + 1) * cell_height  # (num_of_rows) * (height of the cell)
            x_start, x_end = j * cell_width, (j + 1) * cell_width  # (num_of_cols) * (width of the cell)
            cell = gray[y_start: y_end, x_start: x_end]  # slice the original frame into those cells
            cell2 = gray2[y_start: y_end, x_start: x_end]  # slice the new frame into those cells
            if cell.size == 0:
                continue

            laplacian_orig = float(cv2.Laplacian(cell, cv2.CV_64F).var())
            laplacian_new = float(cv2.Laplacian(cell2, cv2.CV_64F).var())
            rel_lap = (laplacian_orig - laplacian_new) / max(laplacian_orig, 1e-6)

            fft_score_orig = float(fft_sharpness(cell))
            fft_score_new = float(fft_sharpness(cell2))
            rel_fft_score = (fft_score_orig - fft_score_new) / max(fft_score_orig, 1e-6)

            ten_score_orig = float(tenengrad(cell))
            ten_score_new = float(tenengrad(cell2))
            rel_ten_score = (ten_score_orig - ten_score_new) / max(ten_score_orig, 1e-6)

            relative_blur = (rel_lap + rel_fft_score + rel_ten_score) / 3.0

            color = (0, 255, 0)  # color cell green if not blurry
            if relative_blur > (
                    rel_threshold / 100):  # mark blurry if both laplacian and fourier transform values are low
                color = (0, 0, 255)  # otherwise its red
                num_blurry += 1
            rect = (x_start, y_start, x_end, y_end, color)
            blur_map.append((rect, rel_lap, rel_fft_score, rel_ten_score, relative_blur))

    final_blur_percentage = (num_blurry / total_cells) * 100

    return final_blur_percentage > blur_percentage, blur_map