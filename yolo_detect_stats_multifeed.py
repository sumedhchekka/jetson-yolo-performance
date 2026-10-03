import os
import sys
import argparse
import time
import csv
import subprocess
import threading

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# =========================================================
# ARGUMENTS
# =========================================================

parser = argparse.ArgumentParser(
    description="Concurrent multi-feed YOLO latency experiment"
)

parser.add_argument(
    '--model',
    help='Path to YOLO model file',
    required=True
)

parser.add_argument(
    '--source',
    help='Video sources. Example: video1.mp4 video2.mp4',
    nargs='+',
    required=True
)

parser.add_argument(
    '--thresh',
    help='Minimum confidence threshold',
    default=0.5
)

parser.add_argument(
    '--resolution',
    help='Resolution in WxH, example: 1280x720',
    default=None
)

args = parser.parse_args()


# =========================================================
# USER SETTINGS
# =========================================================

model_path = args.model
sources = args.source
min_thresh = float(args.thresh)
user_res = args.resolution

EXPERIMENT_DURATION = 60
TEGRASTATS_INTERVAL = 1000
WARMUP_FRAMES = 50


# =========================================================
# CHECK MODEL
# =========================================================

if not os.path.exists(model_path):

    print("ERROR: Model path is invalid or model was not found.")
    sys.exit(1)


# =========================================================
# CHECK VIDEO SOURCES
# =========================================================

vid_ext_list = [
    '.avi',
    '.mov',
    '.mp4',
    '.mkv',
    '.wmv'
]

for source in sources:

    if not os.path.isfile(source):

        print(f"ERROR: Video file not found: {source}")
        sys.exit(1)

    _, ext = os.path.splitext(source)

    if ext.lower() not in vid_ext_list:

        print(f"ERROR: Unsupported video format: {source}")
        sys.exit(1)


# =========================================================
# RESOLUTION
# =========================================================

resize = False

if user_res:

    try:

        resW = int(user_res.split('x')[0])
        resH = int(user_res.split('x')[1])

        resize = True

    except Exception:

        print(
            "ERROR: Resolution must be written as WxH."
        )

        print(
            "Example: --resolution=1280x720"
        )

        sys.exit(1)


# =========================================================
# GLOBAL EXPERIMENT VARIABLES
# =========================================================

experiment_start = None

# Event used to start every feed simultaneously
start_event = threading.Event()

# Event used to stop every feed
stop_event = threading.Event()

# Event used to make sure every feed is ready
ready_event = threading.Event()

results_lock = threading.Lock()

display_frames = {}

thread_errors = {}

# Number of feeds that have successfully initialized
ready_count = 0

ready_count_lock = threading.Lock()


# =========================================================
# TEGRASTATS VARIABLES
# =========================================================

tegrastats_latest = None

tegrastats_lock = threading.Lock()

tegrastats_running = True


# =========================================================
# TEGRASTATS READER
# =========================================================

def read_tegrastats():

    global tegrastats_latest
    global tegrastats_running

    try:

        process = subprocess.Popen(

            [
                "tegrastats",
                "--interval",
                str(TEGRASTATS_INTERVAL)
            ],

            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        while tegrastats_running:

            line = process.stdout.readline()

            if not line:
                break

            line = line.strip()

            if line:

                with tegrastats_lock:

                    tegrastats_latest = line

        process.terminate()

    except Exception as e:

        print()
        print(
            f"WARNING: Unable to start tegrastats: {e}"
        )
        print()


# =========================================================
# PRINT TEGRASTATS
# =========================================================

def print_tegrastats(phase, elapsed):

    print()
    print("======================================================")
    print(f"TEGRASTATS — {phase}")
    print(
        f"Experiment time: {elapsed:.1f} seconds"
    )
    print("======================================================")

    with tegrastats_lock:

        data = tegrastats_latest

    if data:

        print(data)

    else:

        print("No tegrastats data available.")

    print("======================================================")
    print()


# =========================================================
# START TEGRASTATS
# =========================================================

print()
print("Starting tegrastats monitor...")

tegrastats_thread = threading.Thread(
    target=read_tegrastats,
    daemon=True
)

tegrastats_thread.start()

# Give tegrastats time to produce its first reading
time.sleep(2)


# =========================================================
# EXPERIMENT INFORMATION
# =========================================================

print()
print("======================================================")
print("YOLO 60-SECOND CONCURRENT MULTI-FEED EXPERIMENT")
print("======================================================")

print(
    f"Number of feeds: {len(sources)}"
)

print(
    f"Experiment duration: {EXPERIMENT_DURATION} seconds"
)

print(
    f"Warm-up frames per feed: {WARMUP_FRAMES}"
)

print(
    f"Model: {model_path}"
)

print(
    f"Resolution: {user_res if user_res else 'Original'}"
)

print(
    f"Confidence threshold: {min_thresh}"
)

print()
print("0 seconds  = BEFORE")
print("30 seconds = DURING")
print("60 seconds = AFTER")

print("======================================================")
print()


# =========================================================
# FEED WORKER
# =========================================================

def run_feed(feed_id, source):

    global experiment_start
    global ready_count

    cap = None

    try:

        # -------------------------------------------------
        # LOAD MODEL
        # -------------------------------------------------

        print(
            f"[Feed {feed_id}] Loading YOLO model..."
        )

        model = YOLO(
            model_path,
            task='detect'
        )

        labels = model.names


        # -------------------------------------------------
        # OPEN VIDEO
        # -------------------------------------------------

        print(
            f"[Feed {feed_id}] Opening video..."
        )

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():

            raise RuntimeError(
                f"Unable to open video: {source}"
            )


        # -------------------------------------------------
        # VIDEO RESOLUTION
        # -------------------------------------------------

        if user_res:

            cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                resW
            )

            cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                resH
            )


        # -------------------------------------------------
        # LATENCY STORAGE
        # -------------------------------------------------

        inference_latency_buffer = []

        pipeline_latency_buffer = []

        frame_rate_buffer = []

        fps_avg_len = 200

        total_frame_count = 0

        measured_frame_count = 0


        # -------------------------------------------------
        # WARMUP
        # -------------------------------------------------

        print(
            f"[Feed {feed_id}] Performing warm-up..."
        )

        for _ in range(WARMUP_FRAMES):

            ret, frame = cap.read()

            if not ret:

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0
                )

                ret, frame = cap.read()

            if not ret:

                raise RuntimeError(
                    f"Unable to read video during warm-up: {source}"
                )

            if resize:

                frame = cv2.resize(
                    frame,
                    (resW, resH)
                )

            if torch.cuda.is_available():

                torch.cuda.synchronize()

            model(
                frame,
                verbose=False
            )

            if torch.cuda.is_available():

                torch.cuda.synchronize()


        # -------------------------------------------------
        # READY
        # -------------------------------------------------

        with ready_count_lock:

            ready_count += 1

            print(
                f"[Feed {feed_id}] Ready "
                f"({ready_count}/{len(sources)})"
            )

            if ready_count == len(sources):

                ready_event.set()


        # -------------------------------------------------
        # WAIT FOR ALL FEEDS TO START
        # -------------------------------------------------

        start_event.wait()


        # -------------------------------------------------
        # START TIMING
        # -------------------------------------------------

        feed_start = time.perf_counter()


        # -------------------------------------------------
        # MAIN FEED LOOP
        # -------------------------------------------------

        while not stop_event.is_set():

            elapsed_time = (

                time.perf_counter()
                - experiment_start

            )


            if elapsed_time >= EXPERIMENT_DURATION:

                break


            # ---------------------------------------------
            # PIPELINE TIMER
            # ---------------------------------------------

            t_start = time.perf_counter()


            # ---------------------------------------------
            # READ FRAME
            # ---------------------------------------------

            ret, frame = cap.read()


            if not ret:

                # Restart video at beginning

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0
                )

                ret, frame = cap.read()


                if not ret:

                    raise RuntimeError(
                        f"Unable to read video: {source}"
                    )


            total_frame_count += 1


            # ---------------------------------------------
            # RESIZE
            # ---------------------------------------------

            if resize:

                frame = cv2.resize(
                    frame,
                    (resW, resH)
                )


            # ---------------------------------------------
            # CUDA SYNCHRONIZATION
            # ---------------------------------------------

            if torch.cuda.is_available():

                torch.cuda.synchronize()


            t_inference_start = time.perf_counter()


            # ---------------------------------------------
            # YOLO INFERENCE
            # ---------------------------------------------

            results = model(
                frame,
                verbose=False
            )


            if torch.cuda.is_available():

                torch.cuda.synchronize()


            t_inference_stop = time.perf_counter()


            # ---------------------------------------------
            # INFERENCE LATENCY
            # ---------------------------------------------

            inference_latency = (

                t_inference_stop
                - t_inference_start

            ) * 1000


            # ---------------------------------------------
            # PROCESS DETECTIONS
            # ---------------------------------------------

            detections = results[0].boxes

            object_count = 0


            for i in range(len(detections)):

                xyxy_tensor = (
                    detections[i].xyxy.cpu()
                )

                xyxy = (
                    xyxy_tensor
                    .numpy()
                    .squeeze()
                )


                xmin, ymin, xmax, ymax = (
                    xyxy.astype(int)
                )


                classidx = int(
                    detections[i].cls.item()
                )


                classname = labels[classidx]


                conf = detections[i].conf.item()


                if conf > min_thresh:

                    cv2.rectangle(

                        frame,

                        (xmin, ymin),

                        (xmax, ymax),

                        (0, 255, 0),

                        2

                    )


                    label = (

                        f'{classname}: '
                        f'{int(conf * 100)}%'

                    )


                    cv2.putText(

                        frame,

                        label,

                        (
                            xmin,
                            max(ymin - 10, 20)
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.5,

                        (0, 255, 0),

                        1

                    )


                    object_count += 1


            # ---------------------------------------------
            # PIPELINE LATENCY
            # ---------------------------------------------

            t_pipeline_stop = time.perf_counter()


            pipeline_latency = (

                t_pipeline_stop
                - t_start

            ) * 1000


            # ---------------------------------------------
            # STORE MEASUREMENTS
            # ---------------------------------------------

            inference_latency_buffer.append(
                inference_latency
            )

            pipeline_latency_buffer.append(
                pipeline_latency
            )

            measured_frame_count += 1


            # ---------------------------------------------
            # FPS
            # ---------------------------------------------

            if pipeline_latency > 0:

                frame_rate_calc = (
                    1000 / pipeline_latency
                )

            else:

                frame_rate_calc = 0


            if len(frame_rate_buffer) >= fps_avg_len:

                frame_rate_buffer.pop(0)


            frame_rate_buffer.append(
                frame_rate_calc
            )


            avg_frame_rate = np.mean(
                frame_rate_buffer
            )


            # ---------------------------------------------
            # DISPLAY INFORMATION
            # ---------------------------------------------

            cv2.putText(

                frame,

                f'Feed {feed_id}',

                (10, 25),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 255),

                2

            )


            cv2.putText(

                frame,

                f'FPS: {avg_frame_rate:.2f}',

                (10, 50),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 255, 255),

                2

            )


            cv2.putText(

                frame,

                f'Inference: {inference_latency:.2f} ms',

                (10, 75),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 255, 255),

                2

            )


            cv2.putText(

                frame,

                f'Objects: {object_count}',

                (10, 100),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 255, 255),

                2

            )


            cv2.putText(

                frame,

                f'Time: {elapsed_time:.1f}s / 60s',

                (10, 125),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 255, 255),

                2

            )


            # ---------------------------------------------
            # UPDATE DISPLAY FRAME
            # ---------------------------------------------

            with results_lock:

                display_frames[feed_id] = frame.copy()


        # -------------------------------------------------
        # CLEANUP VIDEO
        # -------------------------------------------------

        cap.release()

        cap = None


        # -------------------------------------------------
        # CALCULATE RESULTS
        # -------------------------------------------------

        if measured_frame_count > 0:

            mean_inference = np.mean(
                inference_latency_buffer
            )

            median_inference = np.median(
                inference_latency_buffer
            )

            p95_inference = np.percentile(
                inference_latency_buffer,
                95
            )

            min_inference = np.min(
                inference_latency_buffer
            )

            max_inference = np.max(
                inference_latency_buffer
            )


            mean_pipeline = np.mean(
                pipeline_latency_buffer
            )

            median_pipeline = np.median(
                pipeline_latency_buffer
            )

            p95_pipeline = np.percentile(
                pipeline_latency_buffer,
                95
            )

            average_fps = (
                1000 / mean_pipeline
            )


            # -------------------------------------------------
            # PRINT RESULTS
            # -------------------------------------------------

            print()
            print("======================================================")
            print(f"FEED {feed_id} RESULTS")
            print("======================================================")

            print()
            print("INFERENCE LATENCY")
            print("----------------------------------------------")

            print(
                f"Mean:              {mean_inference:.2f} ms"
            )

            print(
                f"Median:            {median_inference:.2f} ms"
            )

            print(
                f"95th percentile:   {p95_inference:.2f} ms"
            )

            print(
                f"Minimum:           {min_inference:.2f} ms"
            )

            print(
                f"Maximum:           {max_inference:.2f} ms"
            )


            print()
            print("PIPELINE PERFORMANCE")
            print("----------------------------------------------")

            print(
                f"Mean pipeline:     {mean_pipeline:.2f} ms"
            )

            print(
                f"Median pipeline:   {median_pipeline:.2f} ms"
            )

            print(
                f"95th percentile:   {p95_pipeline:.2f} ms"
            )

            print(
                f"Average FPS:       {average_fps:.2f}"
            )


            # -------------------------------------------------
            # SAVE CSV
            # -------------------------------------------------

            csv_filename = (
                f'yolo_feed{feed_id}_latency.csv'
            )


            with open(
                csv_filename,
                'w',
                newline=''
            ) as file:

                writer = csv.writer(file)


                writer.writerow(

                    [
                        'Frame',
                        'Inference Latency (ms)',
                        'Pipeline Latency (ms)'
                    ]

                )


                for i in range(
                    measured_frame_count
                ):

                    writer.writerow(

                        [
                            i + 1,
                            inference_latency_buffer[i],
                            pipeline_latency_buffer[i]
                        ]

                    )


            print()
            print(
                f"Results saved to: {csv_filename}"
            )


    except Exception as e:

        thread_errors[feed_id] = str(e)

        print()
        print(
            f"[Feed {feed_id}] ERROR: {e}"
        )

        # If one feed fails during initialization,
        # make sure the main thread does not wait forever.
        with ready_count_lock:

            ready_count += 1

            if ready_count == len(sources):

                ready_event.set()


        if cap is not None:

            cap.release()


# =========================================================
# START FEED THREADS
# =========================================================

feed_threads = []


for i, source in enumerate(sources):

    feed_id = i + 1


    thread = threading.Thread(

        target=run_feed,

        args=(feed_id, source),

        daemon=True

    )


    feed_threads.append(thread)

    thread.start()


# =========================================================
# WAIT FOR ALL FEEDS TO INITIALIZE
# =========================================================

print()
print("Waiting for all feeds to initialize...")


# Wait until every feed has loaded its model and opened
# its video.
ready_event.wait()


# Give CUDA/TensorRT a short moment to settle
time.sleep(2)


# =========================================================
# CHECK INITIALIZATION ERRORS
# =========================================================

if thread_errors:

    print()
    print("======================================================")
    print("INITIALIZATION ERROR")
    print("======================================================")

    for feed_id, error in thread_errors.items():

        print(
            f"Feed {feed_id}: {error}"
        )

    stop_event.set()

    tegrastats_running = False

    for thread in feed_threads:

        thread.join()

    cv2.destroyAllWindows()

    sys.exit(1)


# =========================================================
# BEFORE MEASUREMENT
# =========================================================

print_tegrastats(
    "BEFORE (0s)",
    0
)


# =========================================================
# START ALL FEEDS SIMULTANEOUSLY
# =========================================================

experiment_start = time.perf_counter()

start_event.set()


print()
print("======================================================")
print("ALL FEEDS STARTED SIMULTANEOUSLY")
print("======================================================")
print()


# =========================================================
# MAIN DISPLAY / EXPERIMENT TIMER
# =========================================================

during_printed = False


while True:

    elapsed_time = (

        time.perf_counter()
        - experiment_start

    )


    # -----------------------------------------------------
    # DURING — 30 SECONDS
    # -----------------------------------------------------

    if (

        elapsed_time >= 30
        and not during_printed

    ):

        print_tegrastats(
            "DURING (30s)",
            elapsed_time
        )

        during_printed = True


    # -----------------------------------------------------
    # STOP AT 60 SECONDS
    # -----------------------------------------------------

    if elapsed_time >= EXPERIMENT_DURATION:

        stop_event.set()

        break


    # -----------------------------------------------------
    # DISPLAY ALL FEEDS
    # -----------------------------------------------------

    with results_lock:

        current_frames = dict(
            display_frames
        )


    for feed_id, frame in current_frames.items():

        cv2.imshow(
            f'YOLO Feed {feed_id}',
            frame
        )


    # -----------------------------------------------------
    # KEYBOARD
    # -----------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    if key == ord('q') or key == ord('Q'):

        print()
        print("Experiment manually stopped.")

        stop_event.set()

        break


# =========================================================
# WAIT FOR FEED THREADS
# =========================================================

for thread in feed_threads:

    thread.join()


# =========================================================
# 60 SECOND MARK
# =========================================================

actual_elapsed = (

    time.perf_counter()
    - experiment_start

)


print()
print("======================================================")
print("60-SECOND EXPERIMENT REACHED")
print("======================================================")


# Allow tegrastats to update
time.sleep(1)


# =========================================================
# AFTER — 60 SECONDS
# =========================================================

print_tegrastats(
    "AFTER (60s)",
    actual_elapsed
)


# =========================================================
# STOP TEGRASTATS
# =========================================================

tegrastats_running = False


# =========================================================
# CLEANUP
# =========================================================

cv2.destroyAllWindows()


# =========================================================
# ERROR REPORT
# =========================================================

if thread_errors:

    print()
    print("======================================================")
    print("FEED ERRORS")
    print("======================================================")

    for feed_id, error in thread_errors.items():

        print(
            f"Feed {feed_id}: {error}"
        )


# =========================================================
# COMPLETE
# =========================================================

print()
print("======================================================")
print("MULTI-FEED EXPERIMENT COMPLETE")
print("======================================================")
