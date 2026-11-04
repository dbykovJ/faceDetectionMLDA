#!/bin/bash

echo "base image with linear scales"
python main.py images/image1.jpeg \
    --out image1_base_linear_scale.jpg \
    -m Net -w Net.pt \
    -c 0 \
    --knn 0 \
    --thickness 4 \
    --scales 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1

echo "fixing scaling"
python main.py images/image1.jpeg \
    --out image1_base_geom_scale.jpg \
    -m Net -w Net.pt \
    -c 0 \
    --knn 0 \
    --thickness 4


echo "filtering certainty 1"
python main.py images/image1.jpeg \
    --out image1_base_c1.jpg \
    -m Net -w Net.pt \
    -c 1 \
    --knn 0 \
    --thickness 4

echo "filtering certainty 3"
python main.py images/image1.jpeg \
    --out image1_base_c3.jpg \
    -m Net -w Net.pt \
    -c 3 \
    --knn 0 \
    --thickness 4


echo "filtering neighbouring k=10"
python main.py images/image1.jpeg \
    --out image1_base_k10.jpg \
    -m Net -w Net.pt \
    -c 0 \
    --knn 10 \
    --thickness 4

echo "friends"
python main.py images/image2.jpg \
    --out friends_detected.jpg \
    -m FCNet -w FCNet.pt \
    -c 0.5 \
    --knn 0 \
    --thickness 6

echo "manifestation"
python main.py images/image5.jpg \
    --out manifestation_no_faces.jpg \
    -m FCNet -w FCNet.pt \
    -c 0.5 \
    --knn 0 \
    --thickness 6

python main.py images/image5.jpg \
    --out manifestation_faces.jpg \
    -m FCNet -w FCNet.pt \
    -c 0.5 \
    --knn 0 \
    --thickness 6 \
    --scales 1.0 0.9 0.81 0.729 0.6561 0.59049

