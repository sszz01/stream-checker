import cv2

class FrameAnalyzer:
    def __init__(self, frame):
        self.frame = frame

    @staticmethod
    def is_blurry(frame, threshold, blur_percentage, grid_size=(3,3)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape
        num_rows, num_cols = grid_size
        cell_height = height // num_rows
        cell_width = width // num_cols
        num_blurry = 0
        total_cells = num_rows * num_cols

        for i in range(num_rows):
            for j in range(num_cols):
                cell = gray[i * cell_height : (i + 1) * cell_height, j * cell_width : (j + 1) * cell_width]
                laplacian = cv2.Laplacian(cell, cv2.CV_64F)
                variance = laplacian.var()
                cv2.rectangle(frame, (j * cell_width, i * cell_height), ((j + 1) * cell_width, (i + 1) * cell_height), (255, 0, 0), 2)
                cv2.putText(frame, f"{variance:.2f}", (j * cell_width, i * cell_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                is_blurry_cell = variance < threshold
                if is_blurry_cell:
                    num_blurry += 1

        return (num_blurry / total_cells) * 100 > blur_percentage