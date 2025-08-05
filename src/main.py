import cv2
from frame_analyzer import FrameAnalyzer

STREAM_SRC = "https://www.bloomberg.com/media-manifest/streams/aus.m3u8" # sample stream

def main():
    cap = cv2.VideoCapture(STREAM_SRC)
    if not cap.isOpened():
        print(f"Cannot open video source: {STREAM_SRC}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 30

    frame_delay = 1000 / fps

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or connection lost")
            break

        frame = cv2.resize(frame, (640, 480))

        analyzer = FrameAnalyzer(frame)
        _, blurry, variance = analyzer.is_blurry()

        cv2.putText(frame, f"FPS: {fps}, isBlurry: {blurry}, variance: {variance}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("stream preview", frame)
        if cv2.waitKey(max(1, int(frame_delay))) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
