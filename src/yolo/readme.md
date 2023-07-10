# YOLO

## Paper notes
* Object detection methods (yolo and prior)
    * Sliding window, classify every windows.
    * Region proposals, classify the proposed (most likely to contain objects) regions.
    * YOLO, end to end approach.

##  YOLO predictions
* The prediction from one image is a single tensor.
* Divide the input image to S x S grid.
* The grid where the center of the ground truth bbox belongs, will be responsible to predict the bounding box.
* Each grid cells predicts B bboxes (x, y, w, h) plus the confidence, also the C classes probs.
* Summary, for a single image, the prediction will be S x S x (B * 5 + C)

# TODO
* check if huggingface datasets can speed up image data (just like audio mmap)
* find bottleneck in training

# TODO NOW
* train on imagenet

# Questions
* How to init cnn layers and linear layers with leaky relu?
* How to supply S (yolo image grid) to model construction? 7 is obtained by the result of previous layers, so not from S

# References
* [YOLO paper](https://arxiv.org/abs/1506.02640)
* [YOLO from scratch](https://www.youtube.com/watch?v=n9_XyCGr-MI)
* [Accelerate + WandB blog](https://wandb.ai/gladiator/HF%20Accelerate%20+%20W&B/reports/Hugging-Face-Accelerate-Super-Charged-with-Weights-Biases--VmlldzoyNzk3MDUx?utm_source=docs&utm_medium=docs&utm_campaign=accelerate-docs)