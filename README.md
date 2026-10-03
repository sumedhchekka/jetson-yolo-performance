# YOLO Object Detection Performance on NVIDIA Jetson Orin Nano

This repository contains the modified YOLO object-detection script used in the study examining the effects of concurrent video-stream workload and sustained thermal conditions on object-detection performance on an NVIDIA Jetson Orin Nano.

## Script

`yolo_detect_stats_multifeed.py` is a modified version of an EdjeElectronics YOLO detection script. The modified script supports concurrent video-stream processing and records:

- Inference latency
- Total pipeline latency
- Processing frame rate
- GPU utilization
- GPU temperature

Hardware-level measurements are collected using NVIDIA's `tegrastats` system utility.
