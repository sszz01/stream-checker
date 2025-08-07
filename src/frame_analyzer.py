import cv2

class FrameAnalyzer:
    def __init__(self, frame):
        self.frame = frame

    @staticmethod
    def is_blurry(frame, threshold, blur_percentage, grid_size):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape
        num_rows, num_cols = grid_size
        cell_height = height // num_rows
        cell_width = width // num_cols
        num_blurry = 0
        total_cells = num_rows * num_cols
        blur_map = []

        for i in range(num_rows):
            for j in range(num_cols):
                y_start, y_end = i * cell_height, (i + 1) * cell_height # (num_of_rows) * (height of the cell)
                x_start, x_end = j * cell_width, (j + 1) * cell_width # (num_of_cols) * (width of the cell)
                cell = gray[y_start: y_end, x_start : x_end] # slice the original frame into those cells
                laplacian = cv2.Laplacian(cell, cv2.CV_64F)
                variance = laplacian.var()

                is_blurry_cell = variance < threshold
                color = (0, 255, 0) # color cell green if not blurry
                if is_blurry_cell:
                    color = (0, 0, 255) # otherwise its red
                    num_blurry += 1
                rect = (x_start, y_start, x_end, y_end, color)
                blur_map.append((rect, variance))

        return ((num_blurry / total_cells) * 100) > blur_percentage, blur_map