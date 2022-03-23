import cv2
import numpy as np
from matplotlib import pyplot as plt


def lucas_kanade_method(video_path_original,video_path_output):
    # Read the video
    cap_original = cv2.VideoCapture(video_path_original)
    cap_output = cv2.VideoCapture(video_path_output)

    # Get frame count
    n_frames = int(cap_output.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get width and height of video
    w = int(cap_output.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_output.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get frames per second (fps)
    fps = cap_output.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # cap_result_original = cv2.VideoWriter('results/video_9/video_9_original.avi', fourcc, fps, (w,h))
    # cap_result_output = cv2.VideoWriter('results/video_9/video_9_output.avi', fourcc, fps, (w,h))

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=30, qualityLevel=0.01, minDistance=30, blockSize=3)

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(winSize = (15, 15),maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),)

    # Create random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret_original, old_frame_original = cap_original.read()
    ret_output, old_frame_output = cap_output.read()
    old_gray_original = cv2.cvtColor(old_frame_original, cv2.COLOR_BGR2GRAY)
    old_gray_output = cv2.cvtColor(old_frame_output, cv2.COLOR_BGR2GRAY)
    p0_output = cv2.goodFeaturesToTrack(old_gray_output, mask=None, **feature_params)
    p0_output = np.delete(p0_output,10,axis = 0)
    p0_output = np.delete(p0_output, 13,axis = 0)
    p0_original = p0_output

    # Create a mask image for drawing purposes
    mask_output = np.zeros_like(old_frame_output)
    mask_original = np.zeros_like(old_frame_original)

    euclidian_dists_original = dict()
    euclidian_dists_output = dict()
    x_y_original = dict()
    x_y_output = dict()
    while True:

        # Read new frame
        ret_original, frame_original = cap_original.read()
        ret_output, frame_output = cap_output.read()
        if not ret_original or not ret_output:
            break
        frame_gray_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
        frame_gray_output = cv2.cvtColor(frame_output, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        p1_original, st_original, err_original = cv2.calcOpticalFlowPyrLK(old_gray_original, frame_gray_original, p0_original, None, **lk_params)
        p1_output, st_output, err_output = cv2.calcOpticalFlowPyrLK(old_gray_output, frame_gray_output, p0_output, None, **lk_params)

        # Select good points
        good_new_original = p1_original[st_original == 1]
        good_old_original = p0_original[st_original == 1]
        good_new_output = p1_output[st_output == 1]
        good_old_output = p0_output[st_output == 1]

        for i, (new, old) in enumerate(zip(good_new_original, good_old_original)):
            a, b = new.ravel()
            c, d = old.ravel()
            # Save euclidian distance and x,y.
            if i not in euclidian_dists_original.keys():
                euclidian_dists_original[i] = []
            if i not in x_y_original.keys():
                x_y_original[i] = {'x':[],'y':[]}
            x_y_original[i]['x'].append(c)
            x_y_original[i]['y'].append(d)
            euclidian_dists_original[i].append(np.linalg.norm(np.array((a,b))-np.array((c,d))))
            # Draw the tracks
            mask_original = cv2.line(mask_original, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame_original = cv2.circle(frame_original, (int(a), int(b)), 5, color[i].tolist(), -1)
        # Display the demo
        img_original = cv2.add(frame_original, mask_original)
        # cap_result_original.write(img_original)
        cv2.imshow("frame_original", img_original)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

        for i, (new, old) in enumerate(zip(good_new_output, good_old_output)):
            a, b = new.ravel()
            c, d = old.ravel()
            # Save euclidian distance and x,y.
            if i not in euclidian_dists_output.keys():
                euclidian_dists_output[i] = []
            if i not in x_y_output.keys():
                x_y_output[i] = {'x':[],'y':[]}
            x_y_output[i]['x'].append(c)
            x_y_output[i]['y'].append(d)
            euclidian_dists_output[i].append(np.linalg.norm(np.array((a,b))-np.array((c,d))))
            # Draw the tracks
            mask_output = cv2.line(mask_output, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame_output = cv2.circle(frame_output, (int(a), int(b)), 5, color[i].tolist(), -1)
        # Display the demo
        img_output = cv2.add(frame_output, mask_output)
        # cap_result_output.write(img_output)
        cv2.imshow("frame_output", img_output)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

        # Update the previous frame and previous points
        old_gray_original = frame_gray_original.copy()
        p0_original = good_new_original.reshape(-1, 1, 2)
        old_gray_output = frame_gray_output.copy()
        p0_output = good_new_output.reshape(-1, 1, 2)

    # Calculate and save variances
    variances_original = []
    for v in euclidian_dists_original.values():
        variances_original.append(np.var(v))

    variances_output = []
    for v in euclidian_dists_output.values():
        variances_output.append(np.var(v))

    print('\naverage of the motion variance in original video is: ' + str(sum(variances_original) / len(variances_original)))
    print('average of the motion variance in output video is: ' + str(sum(variances_output) / len(variances_output)))

    # Plot a graph that shows the motion variance per pixel
    x_data = np.arange(1, len(variances_output) + 1, 1)
    plt.plot(x_data, variances_output, label='output')
    plt.plot(x_data, variances_original, label='original')
    ''' Uncomment the next code line to change the scale of Y axis '''
    # plt.ylim(0,2)
    plt.xlabel('Pixels')
    plt.ylabel('Motion variances')
    plt.title('Motion variance per pixel')
    plt.legend()
    plt.show()

    # Plot multiple X_Y in same graph
    for i in range(len(variances_output)):
        if i != 3 and i != 5 and i != 11: #TODO: pick your indices.
            continue
        plt.plot(x_y_output[i]['x'], x_y_output[i]['y'], label='Output - variance is: ' + str(variances_output[i]))
        plt.plot(x_y_original[i]['x'], x_y_original[i]['y'], label='Original - variance is: ' + str(variances_original[i]))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('X_Y for some pixels')
    plt.legend(prop={'size':16})
    plt.show()

    # Plot X_Y in separated graphs
    for i in range(len(variances_output)):
        if i != 3 and i != 5 and i != 11: #TODO: pick your indices.
            continue
        plt.plot(x_y_output[i]['x'], x_y_output[i]['y'], label='Output - variance is: ' + str(variances_output[i]))
        plt.plot(x_y_original[i]['x'], x_y_original[i]['y'], label='Original - variance is: ' + str(variances_original[i]))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('X_Y for specific pixel')
        plt.legend(prop={'size':16})
        plt.show()

    # Plot Euclidean distance of certain pixel per frame graphs
    for i in range(len(variances_output)):
        if i!=0: #TODO: Pick your certain pixel
            continue
        x_data = np.arange(2,n_frames+1,1)
        plt.plot(x_data, euclidian_dists_output[i], label = 'output')
        plt.plot(x_data, euclidian_dists_original[i], label = 'original')
        plt.xlabel('Frame number')
        plt.ylabel('Euclidean distance')
        plt.title('Euclidean distance of certain pixel per frame')
        plt.legend()
        plt.show()

#TODO: Call the lucas_kande_method with path to the original video and then path to the output video.
lucas_kanade_method('', '')
