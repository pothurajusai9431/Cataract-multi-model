# Cataract Detection Models


The cataract detection system was developed using multiple deep learning architectures to classify eye images into **Cataract** and **Normal** categories. Performance of different convolutional neural network (CNN) models was evaluated to identify the most accurate and reliable model.

#### Models Used

* AlexNet
* DeepANN (Artificial Neural Network)
* DeepCNN (Custom Convolutional Neural Network)
* ResNet
* VGG

#### Dataset

* Two classes:

  * Cataract
  * Normal
* Input image size: **128 × 128 RGB**
* Images were resized and normalized before training.
* Data augmentation techniques such as rotation, flipping, and zooming were applied to improve model generalization and reduce overfitting.

#### Preprocessing

* Image resizing to **128 × 128**
* RGB image normalization
* Data augmentation
* Train–Validation–Test dataset split

#### Training

* Loss Function: **Binary Cross-Entropy**
* Optimizer: **Adam**
* Evaluation Metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * Confusion Matrix

#### Objective

The objective of this project was to compare the performance of various deep learning models for automated cataract detection and determine the architecture that provides the best classification accuracy while maintaining good generalization on unseen eye images.

#### Outcome

The trained models successfully distinguished between cataract and normal eye images. Comparative analysis showed differences in accuracy, computational complexity, and training time, helping identify the most suitable model for cataract screening applications.

<img width="806" height="367" alt="image" src="https://github.com/user-attachments/assets/d34dbe8a-7f8a-4830-92a1-16d0219d1378" />

