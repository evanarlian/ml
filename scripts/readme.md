# ImageNet
Publicly available ImageNet (the one with 1000 classes) is about 130GB in size. It is used to pretrain computer vision models. ImageNet comes with high resolution images, and we *mostly* won't need the full resolution to pretrain image models because the common pipeline is to resize to 256 and random cropping to 224x224.

I wanted to try image pretraining **not** on the full ImageNet. Here are some pre-existing smaller ImageNet alternatives:
* [ImageNet mini](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000), the same 1000 classes but only a few images per class, not enough for pretraining.
* [ImageNet100](https://www.kaggle.com/datasets/ambityga/imagenet100), randomly selected 100 classes from the full 1000. All 100 classes are animals, might not be enough in variation.
* [TensorFlow ImageNet resized](https://www.tensorflow.org/datasets/catalog/imagenet_resized), complete ImageNet with image resized to 8x8, 16x16, 32x32, and 64x64. The downsides are the images were center-cropped (may lose edge informations) and 64x64 is not big enough.

The ideal dataset should be ImageNet with all the smaller size resized to 256 while maintaining aspect ratio. Below are some scripts just to to that. The main idea is to use Hugging Face ImageNet1k to download, resize, and upload the dataset to Hugging Face Hub.

Hugging Face is chosen because it has ability to stream the dataset so we do not have to download the whole 130GB before even starting to resize. To run these:
* Log in with Hugging Face credentials.
* Agree to [ImageNet1k](https://huggingface.co/datasets/imagenet-1k) dataset terms.
* Change some params and run below scripts from project root.
* Go to Hugging Face Hub and add info in the readme.
```bash
# resize and save
python scripts/imagenet_hf_resize.py --min_size 256 --streaming

# upload to hf hub
python scripts/imagenet_hf_push.py --folder data/imagenet_1k_resized_256 --username evanarlian
```

Some extra info about the uploaded dataset:
* Smaller in size.
* No need to resize the image during training (to eliminate images under min_size), just crop it.
* No need to convert to RGB (to eliminate L or RGBA mode), because every images will be converted to RGB JPEG.

See example of the uploaded dataset [here](https://huggingface.co/datasets/evanarlian/imagenet_1k_resized_256).
