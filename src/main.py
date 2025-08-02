import cv2

STREAM_SRC = "https://www.bloomberg.com/media-manifest/streams/aus.m3u8" # sample stream

def main():
    cap = cv2.VideoCapture(STREAM_SRC)
    if not cap.isOpened():
        print(f"Cannot open video source: {STREAM_SRC}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 60

    frame_delay = 1000 / fps

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Stream ended or connection lost")
            break

        cv2.imshow("Frame", frame)
        if cv2.waitKey(max(1, int(frame_delay))) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
