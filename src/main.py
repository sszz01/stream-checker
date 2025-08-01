import cv2
import time

STREAM_SRC = "https://www.bloomberg.com/media-manifest/streams/aus.m3u8" # sample stream

def main():
    start_time = time.time()
    cap = cv2.VideoCapture(STREAM_SRC)
    if not cap.isOpened():
        print(f"cannot open video source: {STREAM_SRC}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    wt = 1 / fps if fps > 0 or fps != fps else exit(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Frame", frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
        dt = time.time() - start_time
        if wt - dt > 0:
            time.sleep(wt - dt)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
