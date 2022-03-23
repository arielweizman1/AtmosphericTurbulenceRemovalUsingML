import os
import cv2

videos_folder = '' #TODO: fill your videos folder path

# index to count videos
j = 1

for file in os.listdir(videos_folder):
    # Read input video
    cap = cv2.VideoCapture(videos_folder+file)
    # Get frame number
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Divide the frames into batch of 15
    parts = [(i*15+1, i*15+15) for i in range(int(length/15))]
    # Read first frame
    ret, frame = cap.read()
    # Get height and width
    h, w, _ = frame.shape

    # Create the video writers for each 15 frames
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    #TODO: fill your path instead of <Save_your_video_here> to save your videos
    writers = [cv2.VideoWriter(f"<Save_your_video_here>/video_{j}_{int((start/15)+1)}.avi", fourcc, 25.0, (w, h)) for start, end in parts]

    f = 0
    while ret:
        f += 1
        for i, part in enumerate(parts):
            start, end = part
            if start <= f <= end:
                writers[i].write(frame)
        ret, frame = cap.read()

    # clean up writers
    for writer in writers:
        writer.release()

    # clean up input
    cap.release()
    j += 1

