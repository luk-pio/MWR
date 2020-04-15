import time
import argparse
import cv2
import sys


def main(input, tracker_type, fps_max):
    trackers = {
            'boosting': cv2.TrackerBoosting_create,
            'mil': cv2.TrackerMIL_create,
            'kcf': cv2.TrackerKCF_create,
            'tld': cv2.TrackerTLD_create,
            'medianflow': cv2.TrackerMedianFlow_create,
            'goturn': cv2.TrackerGOTURN_create,
            'mosse': cv2.TrackerMOSSE_create,
            'csrt': cv2.TrackerCSRT_create,
    }

    tracker = trackers[tracker_type]()
    video = cv2.VideoCapture(input)

    if fps_max not in range(1, 120):
        print(
            "The max fps parameter must be within the range 1 - 120. "
            "Aborting...")
        sys.exit()

    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    (H, W) = frame.shape[:2]

    bbox = None

    prev_time = time.time()
    while True:

        # Throttle fps
        time_elapsed = time.time() - prev_time
        if time_elapsed < 1 / fps_max:
            continue

        ok, frame = video.read()
        if not ok:
            break

        if bbox is not None:
            (success, box) = tracker.update(frame)

            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Define the info we want to show on the video
        info = [
                ("Pause: ", "p"),
                ("Resume: ", "Spc or Enter"),
                ("Quit: ", "q"),
                ("Tracker", tracker_type),
        ]
        # Display info on the video
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("p"):
            bbox = cv2.selectROI("Frame", frame, fromCenter=False,
                                 showCrosshair=True)
            tracker = trackers[tracker_type]()
            tracker.init(frame, bbox)
        elif key == ord("q"):
            break
        prev_time = time.time()

    cv2.destroyAllWindows()


def parse_arguments():
    desc = 'Object tracking software for MWR classes'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input', type=str, help='Path to input video')
    parser.add_argument('-f', '--fps', type=int, help='Max fps', default=30)
    parser.add_argument('-t', '--tracker', type=str, default='mil',
                        choices=['boosting', 'mil', 'kcf', 'tld', 'medianflow',
                                 'goturn', 'mosse', 'csrt'],
                        help='''OpenCV object tracking algorithm
                        Can be one of: boosting, mil, kcf, tld, medianflow, 
                        goturn, mosse, csrt.
                        ''')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args.input, args.tracker, args.fps)
