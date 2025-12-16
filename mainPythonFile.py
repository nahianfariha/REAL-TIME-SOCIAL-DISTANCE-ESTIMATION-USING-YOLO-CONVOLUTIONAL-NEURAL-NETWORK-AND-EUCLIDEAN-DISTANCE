# mainFile.py
# Single-file social distancing detector (no package imports required)
# Put this file in the same folder that contains the 'yolo-coco' directory
# Example run:
#   python mainFile.py --input crowded.mp4 --output out.avi --display 1

import os
import argparse
import time
import numpy as np
import cv2
from scipy.spatial import distance as dist
import imutils

# -----------------------
# Configuration (edit if needed)
# -----------------------
USE_GPU = False          # set True if you built OpenCV with CUDA DNN support
MIN_CONF = 0.3           # minimum probability to filter weak detections
NMS_THRESH = 0.3         # non-maxima suppression threshold
MIN_DISTANCE = 75        # minimum pixel distance for social distancing violation
YOLO_DIR = "yolo-coco"   # folder that contains coco.names, yolov3.cfg, yolov3.weights

# -----------------------
# Helper: detect_people
# -----------------------
def detect_people(frame, net, ln, personIdx=0, min_conf=MIN_CONF, nms_thresh=NMS_THRESH):
    """
    Detect people in `frame` using YOLO net and return a list of:
      [(confidence, (startX, startY, endX, endY), (cX, cY)), ...]
    """
    (H, W) = frame.shape[:2]
    # create a blob and perform a forward pass
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    centroids = []
    confidences = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each detection
        for detection in output:
            # detection: [center_x, center_y, width, height, obj_conf, class_scores...]
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter by person class and min confidence
            if classID == personIdx and confidence > min_conf:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # derive top-left corner
                startX = int(centerX - (width / 2))
                startY = int(centerY - (height / 2))
                endX = startX + int(width)
                endY = startY + int(height)

                boxes.append([startX, startY, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    # apply non-maxima suppression to suppress weak, overlapping boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_conf, nms_thresh)

    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            (startX, startY, w, h) = boxes[i]
            endX = startX + w
            endY = startY + h
            cX, cY = centroids[i]
            results.append((confidences[i], (startX, startY, endX, endY), (int(cX), int(cY))))

    return results

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="",
                    help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str, default="",
                    help="path to optional output video file")
    ap.add_argument("-d", "--display", type=int, default=1,
                    help="whether or not output frame should be displayed (1/0)")
    ap.add_argument("--min-distance", type=int, default=MIN_DISTANCE,
                    help="minimum pixel distance for violation")
    args = vars(ap.parse_args())

    video_path = args["input"]
    output_path = args["output"]
    display = args["display"] > 0
    min_distance = args["min_distance"]

    # verify YOLO files exist
    labelsPath = os.path.join(YOLO_DIR, "coco.names")
    configPath = os.path.join(YOLO_DIR, "yolov3.cfg")
    weightsPath = os.path.join(YOLO_DIR, "yolov3.weights")

    for p in (labelsPath, configPath, weightsPath):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required YOLO file not found: {p}")

    # load class labels
    with open(labelsPath, "r") as f:
        LABELS = f.read().strip().split("\n")

    # load YOLO object detector
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    if USE_GPU:
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # determine output layer names
    ln = net.getLayerNames()
    try:
        # OpenCV 3.x / older 4.x
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except:
        # Newer OpenCV 4.x returns simple list of ints
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # open video capture
    if video_path:
        # allow relative or absolute path
        if not os.path.exists(video_path):
            print(f"[ERROR] input video not found: {video_path}")
            return
        print(f"[INFO] opening video file: {video_path}")
        vs = cv2.VideoCapture(video_path)
    else:
        print("[INFO] no input provided, opening webcam...")
        vs = cv2.VideoCapture(0)

    if not vs.isOpened():
        print("[ERROR] could not open video source. Check the path or webcam.")
        return

    writer = None
    frame_width = None
    frame_height = None

    # processing loop
    print("[INFO] starting processing...")
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            print("[INFO] no more frames or unable to read frame.")
            break

        # resize for faster processing and consistent coordinates
        frame = imutils.resize(frame, width=700)
        if frame_width is None:
            (frame_height, frame_width) = frame.shape[:2]

        # detect people
        results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

        violate = set()
        # compute pairwise distances between centroids if >= 2 people
        if len(results) >= 2:
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < min_distance:
                        violate.add(i)
                        violate.add(j)

        # draw bounding boxes and centroids
        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)   # green

            if i in violate:
                color = (0, 0, 255)  # red

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

            # optionally show confidence
            text = f"{prob:.2f}"
            cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # overlay violation count
        info_text = f"Social Distancing Violations: {len(violate)}"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

        # show the output frame
        if display:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # press 'q' to exit
            if key == ord("q"):
                print("[INFO] user requested exit (q).")
                break

        # initialize video writer if output requested
        if output_path != "" and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output_path, fourcc, 25,
                                     (frame.shape[1], frame.shape[0]), True)
            if not writer.isOpened():
                print("[WARNING] Unable to open video writer. Output will not be saved.")
                writer = None

        # write frame to output file if writer available
        if writer is not None:
            writer.write(frame)

    # cleanup
    print("[INFO] cleaning up...")
    if writer is not None:
        writer.release()
    vs.release()
    if display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
