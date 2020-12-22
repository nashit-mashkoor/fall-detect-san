import os
import cv2
import time
import glob
import numpy as np

def record_video(duration=4):
    cam_index = None
    cam_array = []
    for camera in glob.glob("/dev/video?"):
        c = cv2.VideoCapture(camera)
        ret, frame = c.read()
        if frame is not None:
            s = ''.join(x for x in camera if x.isdigit())
            cam_array.append(int(s))
        c.release()
    cam_index = int(np.max(cam_array))
    print(cam_index)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('notfall_14.avi', fourcc, 20.0, (640, 480))

    i = 1
    cap = cv2.VideoCapture(cam_index)
    start_time = time.time()
    while int(time.time() - start_time) <= duration:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # print(frame.shape)
        if ret:
            frame = cv2.flip(frame, 1)
            out.write(frame)
            cv2.imshow('frame', frame)
            print(frame[:, :, 0].shape)
            r = frame.copy()
            r[:, :, 0] = 0
            r[:, :, 1] = 0
            cv2.imshow("Red", r)
            b = frame.copy()
            b[:, :, 2] = 0
            b[:, :, 1] = 0
            cv2.imshow("Blue", b)
            g = frame.copy()
            g[:, :, 2] = 0
            g[:, :, 0] = 0
            cv2.imshow("Green", g)
            cv2.waitKey(1)
            # if i % 3 == 0:
            #     cv2.imwrite(depth + "/" + str(i/3) + ".jpg", frame)
            i+=1
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(i)
    print("Video Captured and saved")


record_video(20)