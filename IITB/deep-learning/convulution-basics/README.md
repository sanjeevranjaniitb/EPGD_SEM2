# Dog vs Cat Convolution + Pooling Demo

This Jupyter Notebook (`dogcat_conv_pool_demo_with_classifier.ipynb`) provides a step-by-step, educational demonstration of fundamental concepts in convolutional neural networks (CNNs), specifically:

* **Convolution:** Applying filters (like Sobel filters for edge detection) to an image.

* **ReLU Activation:** Using the Rectified Linear Unit function to introduce non-linearity.

* **Pooling:** Downsampling feature maps to reduce dimensionality and retain important features.

* **Feature Extraction & Classification:** Using the extracted features to train a simple Logistic Regression classifier to distinguish between images of cats and dogs.

## Prerequisites

* **Jupyter Notebook:** The notebook is designed to be run in a Jupyter environment.

* **Python Libraries:** The notebook requires the following Python libraries. You can install them using pip:

  * `numpy`

  * `matplotlib`

  * `Pillow` (PIL)

  * `scikit-learn`

* **Dataset:** The notebook assumes you have a small dataset of dog and cat images located on your desktop in a folder named `dogcat_demo`. The images should be named `cat1.jpg`, `cat2.jpg`, `dog1.jpg`, and `dog2.jpg`. The code can be easily modified to point to a different location or use different image names.

## How to Use

1. **Clone the Repository:**
2. **Add Images:** Place your cat and dog images in the `dogcat_demo` folder on your desktop, as described in the prerequisites.
3. **Launch Jupyter:** Open the Jupyter Notebook from your terminal:
4. **Run the Notebook:** Open `dogcat_conv_pool_demo_with_classifier.ipynb` and execute the cells in order. The notebook contains explanations and visualizations to guide you through the process.

## About the Classifier

The final part of the notebook demonstrates a simple, end-to-end classification pipeline. It extracts features from a small set of images and uses a **Logistic Regression** model to classify them. It's important to note that this is a **toy demo** and not a robust or production-ready model, as it is trained on a very small dataset for illustrative purposes. The high accuracy on the training data is not a reliable indicator of real-world performance.
