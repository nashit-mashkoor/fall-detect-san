from FallDetection import detect_fall_per_frame_onnx
import cv2
import time

def live_test():

    cam_index = int(detect_fall_per_frame.argss[0].cam_index)
    cam = cv2.VideoCapture(cam_index)
    while True:
        print("------------------------")
        start = time.time()
        _, img = cam.read()
        dict_vis = detect_fall_per_frame.extract_keypoints_parallel(img)
        result_str = detect_fall_per_frame.alg2_sequential(dict_vis)
        print(result_str)
        print("Pred time on frame " + str(detect_fall_per_frame.frame) + ": " + str(time.time() - start))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def video_test():
    filepath = "fall_4.avi"
    cam = cv2.VideoCapture(filepath)
    while cam.isOpened():
        print("------------------------")
        start = time.time()
        _, img = cam.read()
        if img is not None:
            # cv2.imshow('output', img)
            dict_vis = detect_fall_per_frame_onnx.extract_keypoints_parallel(img)
            result_str = detect_fall_per_frame_onnx.alg2_sequential(dict_vis)
            print(result_str)
            print("Pred time on frame " + str(detect_fall_per_frame_onnx.frame) + ": " + str(time.time() - start))
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_test()