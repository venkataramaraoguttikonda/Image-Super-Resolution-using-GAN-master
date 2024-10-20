
# Image Super-Resolution Using GANs

This project implements an Image Super-Resolution model using Generative Adversarial Networks (GANs). The goal is to upscale low-resolution images to high-resolution images, preserving perceptual quality and enhancing fine details. The model uses a Generator to produce high-resolution images and a Discriminator to distinguish between real and generated images.

## Project Structure

- **datautils.py**: Handles data loading and preprocessing.
- **model.py**: Contains the Generator and Discriminator architectures.
- **loss.py**: Defines the loss functions for content, adversarial, and perceptual losses.
- **train.py**: The main training script to train the GAN.
- **test.py**: Script for testing the trained Generator on new images.
- **DIV2K_train_HR/**: Folder containing high-resolution training images.

## Requirements

To install the necessary dependencies, run:

```bash
pip install torch torchvision tqdm Pillow
```

## Dataset

The project uses the DIV2K dataset for training. The high-resolution images in this dataset are downsampled using bicubic interpolation to create low-resolution inputs.

- **HR Images**: High-resolution images are stored in the `DIV2K_train_HR` folder.
- **Low-Resolution Generation**: During training, low-resolution images are generated dynamically by downscaling high-resolution images.

## Model Architecture

### Generator

The Generator is a deep convolutional network designed to upscale images:

- **Convolutional Layers**: For feature extraction.
- **Residual Blocks**: 16 blocks for learning residuals between the low-resolution and high-resolution images.
- **Sub-Pixel Convolution**: For upscaling image dimensions.
- **PReLU Activation**: To introduce non-linearity.

### Discriminator

The Discriminator is a convolutional network that classifies images as either real (high-resolution) or fake (generated):

- **Convolutional Layers**: Extracts features from the input images.
- **Leaky ReLU Activation**: Helps the network learn efficiently.

## Loss Functions

The loss function is a combination of:

- **Content Loss (MSE)**: Measures the difference between the generated and real images in VGG feature space.
- **Adversarial Loss**: Encourages the Generator to produce images that fool the Discriminator.
- **Perceptual Loss**: A weighted sum of content and adversarial losses.

## Training

To train the model, run the following command:

```bash
python train.py --scale_factor 4 --crop_size 80 --epochs 100
```

- `--scale_factor`: The upscaling factor (e.g., 4x).
- `--crop_size`: The size of the image crops during training.
- `--epochs`: Number of training epochs.

The trained Generator and Discriminator models will be saved in the `saving/gen` and `saving/disc` folders, respectively.

## Testing

To test the trained Generator on a low-resolution image, use the following command:

```bash
python test.py --img_path path/to/low-resolution-image.jpg --model_path path/to/generator.pth
```

This will save the upscaled image in the `outputs/` folder.

## Results

The model outputs high-quality images with enhanced resolution. The performance is evaluated using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).

| Model  | PSNR   | SSIM  |
|--------|--------|-------|
| SRGAN  | 31.927 | 0.902 |
| Bicubic | 26.509 | 0.728 |

## Folder Structure

```
├── datautils.py
├── loss.py
├── model.py
├── test.py
├── train.py
├── DIV2K_train_HR/
├── outputs/
├── saving/
│   ├── gen/
│   └── disc/
```

## References

- Ledig, Christian, et al. "Photo-realistic single image super-resolution using a generative adversarial network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
- I. Goodfellow, et al. "Generative Adversarial Nets." Advances in Neural Information Processing Systems (NIPS), 2014.
