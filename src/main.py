import cv2
import math
import time
import logging
from frame_analyzer import FrameAnalyzer
from errors import StreamError

STREAM_SRC = "https://www.bloomberg.com/media-manifest/streams/aus.m3u8" # sample stream
BLUR_THRESHOLD = 300
FPS_RESET_INTERVAL = 1 # num of seconds before fps is calculated

def setup_logging():
    logging.basicConfig(filename="./logs/incidents.txt", filemode="a",
                        format="[%(asctime)s] %(levelname)s - %(message)s", level=logging.INFO)

def main():
    cap = cv2.VideoCapture(STREAM_SRC)
    if not cap.isOpened():
        print(f"Cannot open video source: {STREAM_SRC}")
        return

    start_time = time.time()
    frame_count = 0
    setup_logging()
    last_logtime = time.time()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30

    frame_delay = 1000 / fps

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or connection lost")
            break

        frame = cv2.resize(frame, (640, 480))
        is_blurry, blur_map = FrameAnalyzer.is_blurry(frame, BLUR_THRESHOLD, 75, (5,3))

        current_logtime = time.time()
        if is_blurry and (current_logtime - last_logtime) > 5:
            logging.info(f"INCIDENT TYPE:{StreamError.BLUR.name} DETECTED")
            last_logtime = current_logtime

        y0 = 30
        line_height = 30
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Blurry: {is_blurry}", (10, y0 + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        for rect, variance in blur_map:
            x_start, y_start, x_end, y_end, color = rect
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)
            cv2.putText(frame, f"{variance:.2f}", (x_start, y_start + (y_end - y_start)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color, 2)
        cv2.imshow("stream preview", frame)

        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > FPS_RESET_INTERVAL:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        if cv2.waitKey(max(1, int(frame_delay))) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
