import cv2
import logging
from FallDetection.visual import activity_dict
from FallDetection.processor import Processor
from FallDetection.inv_pendulum import *
from FallDetection.model import LSTMModel
import openpifpaf
import torch
import math
import time
import argparse
import copy
from FallDetection.default_params import *


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # TODO: Verify the args since they were changed in v0.10.0
    openpifpaf.decoder.cli(parser, force_complete_pose=True,
                           instance_threshold=0.2, seed_threshold=0.5)
    openpifpaf.network.cli(parser)
    parser.add_argument('--resolution', default=0.4, type=float,
                        help=('Resolution prescale factor from 640x480. '
                              'Will be rounded to multiples of 16.'))
    parser.add_argument('--resize', default=None, type=str,
                        help=('Force input image resize. '
                              'Example WIDTHxHEIGHT.'))
    parser.add_argument('--num_cams', default=1, type=int,
                        help='Number of Cameras.')
    parser.add_argument('--cam_index', default=None, type=int,
                        help='Camera Index to capture video.')
    parser.add_argument('--video', default=None, type=str,
                        help='Path to the video file.\nFor single video fall detection(--num_cams=1), save your videos as abc.xyz and set --video=abc.xyz\nFor 2 video fall detection(--num_cams=2), save your videos as abc1.xyz & abc2.xyz and set --video=abc.xyz')
    parser.add_argument('--record_video', default=False,
                        help='Whether to get live stream or not')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='debug messages and autoreload')
    parser.add_argument('--disable_cuda', default=False, action='store_true',
                        help='disables cuda support and runs from gpu')

    vis_args = parser.add_argument_group('Visualisation')
    vis_args.add_argument('--plot_graph', default=False, action='store_true',
                          help='Plot the graph of features extracted from keypoints of pose.')
    vis_args.add_argument('--joints', default=True, action='store_true',
                          help='Draw joint\'s keypoints on the output video.')
    vis_args.add_argument('--skeleton', default=True, action='store_true',
                          help='Draw skeleton on the output video.')
    vis_args.add_argument('--coco_points', default=False, action='store_true',
                          help='Visualises the COCO points of the human pose.')
    vis_args.add_argument('--save_output', default=False, action='store_true',
                          help='Save the result in a video file. Output videos are saved in the same directory as input videos with "out" appended at the start of the title')
    vis_args.add_argument('--fps', default=20, type=int,
                          help='FPS for the output video.')

    args = parser.parse_args()

    # Log
    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

    # Add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    if args.checkpoint is None:
        args.checkpoint = 'shufflenetv2k16w'

    return args

SPIN_COUNTER = 0
t0 = time.time()
frame = 0
consecutive_frames = 36
num_matched = 0

args = cli()
argss = [copy.deepcopy(args) for _ in range(args.num_cams)]

model = LSTMModel(h_RNN=32, h_RNN_layers=2, drop_p=0.2, num_classes=7)
model.load_state_dict(torch.load('FallDetection/lstm2.sav', map_location=argss[0].device))
print("Model Loaded")
print("Model loaded in time: " + str(time.time() - t0))

ip_sets = [[] for _ in range(argss[0].num_cams)]
lstm_sets = [[] for _ in range(argss[0].num_cams)]


def resize(img, resize, resolution):
    # Resize the video
    if resize is None:
        height, width = img.shape[:2]
    else:
        width, height = [int(dim) for dim in resize.split('x')]
    width_height = (int(width * resolution // 16) * 16,
                    int(height * resolution // 16) * 16)
    return width, height, width_height

def extract_keypoints_parallel(img):
    # print(args)
    global argss
    args = argss[0]
    global frame
    global t0
    global SPIN_COUNTER
    tagged_df = None
    dict_vis = None
    width, height, width_height = resize(img, args.resize, args.resolution)
    logging.debug(f'Target width and height = {width_height}')
    processor_singleton = Processor(width_height, args)

    # output_video = None
    print('Reading image ' + str(frame+1))
    frame += 1
    # self_counter.value += 1
    if tagged_df is None:
        curr_time = time.time()
    else:
        curr_time = tagged_df.iloc[frame - 1]['TimeStamps'][11:]
        curr_time = sum(x * float(t) for x, t in zip([3600, 60, 1], curr_time.split(":")))
    if img is None:
        print('no more images captured')
        # print(args.video, curr_time, sep=" ")
        return None

    img = cv2.resize(img, (width, height))
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    keypoint_sets, bb_list, width_height = processor_singleton.single_image(img)
    assert bb_list is None or (type(bb_list) == list)
    if bb_list:
        assert type(bb_list[0]) == tuple
        assert type(bb_list[0][0]) == tuple
    # assume bb_list is a of the form [(x1,y1),(x2,y2)),etc.]

    if args.coco_points:
        keypoint_sets = [keypoints.tolist() for keypoints in keypoint_sets]
    else:
        anns = [get_kp(keypoints.tolist()) for keypoints in keypoint_sets]
        ubboxes = [(np.asarray([width, height]) * np.asarray(ann[1])).astype('int32')
                   for ann in anns]
        lbboxes = [(np.asarray([width, height]) * np.asarray(ann[2])).astype('int32')
                   for ann in anns]
        bbox_list = [(np.asarray([width, height]) * np.asarray(box)).astype('int32') for box in bb_list]
        uhist_list = [get_hist(hsv_img, bbox) for bbox in ubboxes]
        lhist_list = [get_hist(img, bbox) for bbox in lbboxes]
        keypoint_sets = [{"keypoints": keyp[0], "up_hist": uh, "lo_hist": lh, "time": curr_time, "box": box}
                         for keyp, uh, lh, box in zip(anns, uhist_list, lhist_list, bbox_list)]

        # cv2.polylines(img, ubboxes, True, (255, 0, 0), 2)
        # cv2.polylines(img, lbboxes, True, (0, 255, 0), 2)
        for box in bbox_list:
            # print(tuple(box[0]))
            # print(tuple(box[1]))
            cv2.rectangle(img, tuple(box[0]), tuple(box[1]), ((0, 0, 255)), 2)
            # cv2.imwrite("boxes/" + str(self.frame) + ".jpg", img)

        dict_vis = {"img": img, "keypoint_sets": keypoint_sets, "width": width, "height": height,
                    "vis_keypoints": args.joints,
                    "vis_skeleton": args.skeleton, "CocoPointsOn": args.coco_points,
                    "tagged_df": {"text": f"Avg FPS: {frame // (time.time() - t0)}, Frame: {frame}",
                                  "color": [0, 0, 0]}}
    # print(dict_vis)
    return dict_vis


###################################################### Post human estimation ###########################################################


def remove_wrongly_matched(matched_1, matched_2):

    unmatched_idxs = []
    i = 0

    for ip1, ip2 in zip(matched_1, matched_2):
        # each of these is a set of the last t framses of each matched person
        correlation = cv2.compareHist(last_valid_hist(ip1)["up_hist"], last_valid_hist(ip2)["up_hist"], cv2.HISTCMP_CORREL)
        if correlation < 0.5*HIST_THRESH:
            unmatched_idxs.append(i)
        i += 1

    return unmatched_idxs


def match_unmatched(unmatched_1, unmatched_2, lstm_set1, lstm_set2):
    global num_matched
    new_matched_1 = []
    new_matched_2 = []
    new_lstm1 = []
    new_lstm2 = []
    final_pairs = [[], []]

    if not unmatched_1 or not unmatched_2:
        return final_pairs, new_matched_1, new_matched_2, new_lstm1, new_lstm2

    correlation_matrix = - np.ones((len(unmatched_1), len(unmatched_2)))
    dist_matrix = np.zeros((len(unmatched_1), len(unmatched_2)))
    for i in range(len(unmatched_1)):
        for j in range(len(unmatched_2)):
            correlation_matrix[i][j] = cv2.compareHist(last_valid_hist(unmatched_1[i])["up_hist"],
                                                       last_valid_hist(unmatched_2[j])["up_hist"], cv2.HISTCMP_CORREL)
            dist_matrix[i][j] = np.sum(np.absolute(last_valid_hist(unmatched_1[i])["up_hist"]-last_valid_hist(unmatched_2[j])["up_hist"]))

    freelist_1 = [i for i in range(len(unmatched_1))]
    pair_21 = [-1]*len(unmatched_2)
    unmatched_1_preferences = np.argsort(-correlation_matrix, axis=1)
    unmatched_indexes1 = [0]*len(unmatched_1)
    finish_array = [False]*len(unmatched_1)
    while freelist_1:
        um1_idx = freelist_1[-1]
        if finish_array[um1_idx] == True:
            freelist_1.pop()
            continue
        next_unasked_2 = unmatched_1_preferences[um1_idx][unmatched_indexes1[um1_idx]]
        if pair_21[next_unasked_2] == -1:
            pair_21[next_unasked_2] = um1_idx
            freelist_1.pop()
        else:
            curr_paired_2 = pair_21[next_unasked_2]
            if correlation_matrix[curr_paired_2][next_unasked_2] < correlation_matrix[um1_idx][next_unasked_2]:
                pair_21[next_unasked_2] = um1_idx
                freelist_1.pop()
                if not finish_array[curr_paired_2]:
                    freelist_1.append(curr_paired_2)

        unmatched_indexes1[um1_idx] += 1
        if unmatched_indexes1[um1_idx] == len(unmatched_2):
            finish_array[um1_idx] = True

    for j, i in enumerate(pair_21):
        if correlation_matrix[i][j] > HIST_THRESH:
            final_pairs[0].append(i+num_matched)
            final_pairs[1].append(j+num_matched)
            new_matched_1.append(unmatched_1[i])
            new_matched_2.append(unmatched_2[j])
            new_lstm1.append(lstm_set1[i])
            new_lstm2.append(lstm_set2[j])

    # print("finalpairs", final_pairs, sep="\n")

    return final_pairs, new_matched_1, new_matched_2, new_lstm1, new_lstm2


def alg2_sequential(dict_vis):
    global argss
    global consecutive_frames
    args = argss[0]
    global ip_sets
    global lstm_sets
    global num_matched
    global frame
    max_length_mat = 300
    new_num = None
    result_str = None
    indxs_unmatched = None
    result = None
    if not args.plot_graph:
        max_length_mat = consecutive_frames
    if dict_vis is not None:
        dict_frames = [dict_vis]
        # print("Len of dict_frames: " + str(dict_frames))
        kp_frames = [dict_frame["keypoint_sets"] for dict_frame in dict_frames]
        # print(kp_frames)
        if args.num_cams == 1:
            num_matched, new_num, indxs_unmatched = match_ip(ip_sets[0], kp_frames[0], lstm_sets[0], num_matched, max_length_mat)
            valid1_idxs, prediction = get_all_features(ip_sets[0], lstm_sets[0])
            if prediction != 16:
                dict_frames[0]["tagged_df"]["text"] += f" Pred: {activity_dict[prediction+5]}"
                result_str = "Prediction on frame " + str(frame) + ": " + str(activity_dict[prediction+5])
                # print("Prediction on frame " + str(frame) + ": " + str(activity_dict[prediction+5]))
                # result = activity_dict[prediction+5]
            # print(prediction)
    # else:
    #     print("Dict_vis is none")
    return result_str


def get_all_features(ip_set, lstm_set):
    global model
    valid_idxs = []
    invalid_idxs = []
    predictions = [15]*int(len(ip_set))  # 15 is the tag for None

    for i, ips in enumerate(ip_set):
        last1 = None
        last2 = None
        for j in range(-2, -1*DEFAULT_CONSEC_FRAMES - 1, -1):
            if ips[j] is not None:
                if last1 is None:
                    last1 = j
                elif last2 is None:
                    last2 = j
        if ips[-1] is None:
            invalid_idxs.append(i)
        else:
            ips[-1]["features"] = {}
            # get re, gf, angle, bounding box ratio, ratio derivative
            ips[-1]["features"]["height_bbox"] = get_height_bbox(ips[-1])
            ips[-1]["features"]["ratio_bbox"] = FEATURE_SCALAR["ratio_bbox"]*get_ratio_bbox(ips[-1])

            body_vector = ips[-1]["keypoints"]["N"] - ips[-1]["keypoints"]["B"]
            ips[-1]["features"]["angle_vertical"] = FEATURE_SCALAR["angle_vertical"]*get_angle_vertical(body_vector)
            ips[-1]["features"]["log_angle"] = FEATURE_SCALAR["log_angle"]*np.log(1 + np.abs(ips[-1]["features"]["angle_vertical"]))

            if last1 is None:
                invalid_idxs.append(i)
            else:
                ips[-1]["features"]["re"] = FEATURE_SCALAR["re"]*get_rot_energy(ips[last1], ips[-1])
                ips[-1]["features"]["ratio_derivative"] = FEATURE_SCALAR["ratio_derivative"]*get_ratio_derivative(ips[last1], ips[-1])
                if last2 is None:
                    invalid_idxs.append(i)
                    # continue
                else:
                    ips[-1]["features"]["gf"] = get_gf(ips[last2], ips[last1], ips[-1])
                    valid_idxs.append(i)

        xdata = []
        if ips[-1] is None:
            for feat in FEATURE_LIST[:FRAME_FEATURES]:
                if last1 == None: # In case of error
                    return [], 16
                xdata.append(ips[last1]["features"][feat])
            xdata += [0]*(len(FEATURE_LIST)-FRAME_FEATURES)
        else:
            for feat in FEATURE_LIST:
                if feat in ips[-1]["features"]:
                    xdata.append(ips[-1]["features"][feat])
                else:
                    xdata.append(0)

        xdata = torch.Tensor(xdata).view(-1, 1, 5)
        previous_lstm = lstm_set[i][0]

        
        outputs, lstm_set[i][0] = model(xdata, lstm_set[i][0])


        # print(outputs)
        # print(type(outputs))
        # print(outputs.shape)
        # print(lstm_set[i][0])
        # print(type(lstm_set[i][0]))
        # print(lstm_set[i][0][0].shape,lstm_set[i][0][1].shape )
        if np.array_equal(previous_lstm, lstm_set[i][0]):
            print(True)
        else:
            print(False)
        # print('lstm input x_data:')
        # print(type(xdata))
        # print(xdata.shape)
        # print("lstm_set shape:")
        # print(np.array(lstm_set[i][0]).shape)
        # print(type(lstm_set[i][0]))
        if i == 0:
            outval = torch.max(outputs.data, 1)[1][0].item()
            # print(outval)
            if outval == 1:
                prediction = outval
            else:
                indices = np.argsort(np.array(outputs.data))
                if indices[0][6] == 0 or indices[0][5] == 0:
                    prediction = 0
                else:
                    prediction = torch.max(outputs.data, 1)[1][0].item()
            # print("outval: " + str(outval))
            if prediction in [1, 2, 3, 5]:
                lstm_set[i][3] = 0
                if lstm_set[i][2] < EMA_FRAMES:
                    if ips[-1] is not None:
                        lstm_set[i][2] += 1
                        lstm_set[i][1] = (lstm_set[i][1]*(lstm_set[i][2]-1) + get_height_bbox(ips[-1]))/lstm_set[i][2]
                else:
                    if ips[-1] is not None:
                        lstm_set[i][1] = (1-EMA_BETA)*get_height_bbox(ips[-1]) + EMA_BETA*lstm_set[i][1]

            elif prediction == 0:
                if ips[-1] is not None and lstm_set[i][1] != 0 and \
                        abs(ips[-1]["features"]["angle_vertical"]) < math.pi/4:
                    prediction = 7
                else:
                    lstm_set[i][3] += 1
                    if lstm_set[i][3] < DEFAULT_CONSEC_FRAMES//4:
                        prediction = 7
            else:
                lstm_set[i][3] = 0
            predictions[i] = prediction

    return valid_idxs, predictions[0] if len(predictions) > 0 else 15


def get_frame_features(ip_set, new_frame, re_matrix, gf_matrix, max_length_mat=DEFAULT_CONSEC_FRAMES):
    global num_matched
    match_ip(ip_set, new_frame, re_matrix, gf_matrix, max_length_mat)
    return
    for i in range(len(ip_set)):
        if ip_set[i][-1] is not None:
            if ip_set[i][-2] is not None:
                pop_and_add(re_matrix[i], get_rot_energy(
                            ip_set[i][-2], ip_set[i][-1]), max_length_mat)
            elif ip_set[i][-3] is not None:
                pop_and_add(re_matrix[i], get_rot_energy(
                            ip_set[i][-3], ip_set[i][-1]), max_length_mat)
            elif ip_set[i][-4] is not None:
                pop_and_add(re_matrix[i], get_rot_energy(
                            ip_set[i][-4], ip_set[i][-1]), max_length_mat)
            else:
                pop_and_add(re_matrix[i], 0, max_length_mat)
        else:
            pop_and_add(re_matrix[i], 0, max_length_mat)

    for i in range(len(ip_set)):
        if ip_set[i][-1] is not None:
            last1 = None
            last2 = None
            for j in [-2, -3, -4, -5]:
                if ip_set[i][j] is not None:
                    if last1 is None:
                        last1 = j
                    elif last2 is None:
                        last2 = j

            if last2 is None:
                pop_and_add(gf_matrix[i], 0, max_length_mat)
                continue

            pop_and_add(gf_matrix[i], get_gf(ip_set[i][last2], ip_set[i][last1],
                                             ip_set[i][-1]), max_length_mat)

        else:

            pop_and_add(gf_matrix[i], 0, max_length_mat)

    return

def check_consec_frames_pred(match_str, result, consec_num):
    consec_frames = True
    (values, counts) = np.unique(result, return_counts=True)
    if match_str in values:
        match_str_count = counts[np.where(values == match_str)]
        while not consec_frames:
            if match_str_count >= consec_num:
                try:
                    index = result.index(match_str)
                    for i in range(index, index + consec_num):
                        if result[i] != match_str:
                            consec_frames = False
                    if not consec_frames:
                        index = result.index(match_str, index + consec_num)
                except ValueError:
                    consec_frames = False
                    break
