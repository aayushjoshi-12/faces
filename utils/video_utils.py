import cv2 as cv

def read_video(video_path):
    cap = cv.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, output_path):
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter(output_path, fourcc, 30, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved at {output_path}")

def record_video(output_path):
    cap = cv.VideoCapture(0)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        cv.imshow("frame", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    save_video(frames, output_path)    