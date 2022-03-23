# Import the necessary packages
from pyimagesearch.blur_detector import detect_blur_fft
import argparse
import imutils
import time
import cv2
from PIL import Image

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--thresh", type=int, default=5, help="threshold for our blur detector to fire")
args = vars(ap.parse_args())

# Read input video
print("[INFO] starting video stream...")
video_path = '' #TODO: fill your video path
cap = cv2.VideoCapture(video_path)
time.sleep(2.0)
# Loop over the frames from the video
images = []
means = []
i = 0
while True:
    # Grab the frame from the video and resize it to have a maximum width of 500 pixels
    success, frame = cap.read()
    if not success:
        break
    i += 1
    frame = imutils.resize(frame, width=500)
    # Convert the frame to grayscale and detect blur in it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (mean, blurry) = detect_blur_fft(gray, size=60, thresh=args["thresh"])
    means.append(mean)
    # Draw on the frame, indicating whether or not it is blurry
    color = (0, 0, 255) if blurry else (0, 255, 0)
    text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
    text = text.format(mean)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)

    # Save the output frame
    image = Image.fromarray(frame)
    save_image = '' #TODO : fill here the path where to save the frame, use i for uniqueness
    image.save(save_image)
    images.append(save_image)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Create an output video with text on each frame
height, width, _ = cv2.imread(images[0]).shape
fourcc = cv2.VideoWriter_fourcc(*"XVID")
#TODO: fill your video path to save the video instead of <Your_path>
out = cv2.VideoWriter('<Your_path>', fourcc, 25, (width, height))
for image in images:
    out.write(cv2.imread(image))

# Print the means average
print(str(sum(means)/len(means)))

# Do a bit of cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

