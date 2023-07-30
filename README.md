# Facial_Emotion_Detection_Using_CNN
**Facial Emotion Detection using CNN (Convolutional Neural Networks)**

![Facial Emotion Detection](https://your-image-url.com)

## Overview
This repository contains the code and resources for building a facial emotion detection model using Convolutional Neural Networks (CNNs). The model is trained on the FER-2013 dataset, which consists of facial images labeled with seven emotion classes: angry, disgust, fear, happy, sad, surprise, and neutral. The trained model achieves an accuracy of 85.36% on the test set, showcasing its effectiveness in recognizing emotions from facial expressions.

## FER-2013 Dataset
The FER-2013 dataset was downloaded from Kaggle and is widely used in facial emotion recognition tasks. It contains a total of 35,887 images, split into two folders: "train" and "test." Each image is associated with one of the seven emotion classes, making it an ideal dataset for training and evaluating emotion detection models.

## CNN Model Architecture
The facial emotion detection model is implemented using a CNN architecture. The convolutional layers serve to extract relevant features from the facial images, while the fully connected layers aid in the classification process. The architecture is designed to learn expressive patterns from the dataset, enabling it to discern emotions from facial expressions effectively.

## Training and Testing
The training of the CNN model was performed on the entire "train" dataset. Since the FER-2013 dataset comes pre-split into training and testing sets, no further data splitting was required. After training, the model achieved an impressive accuracy of 85.36% on the test set, demonstrating its ability to generalize to new, unseen data.

## Model Storage
Once the model training was completed, the trained CNN model was saved in the "model" folder. This facilitates easy access to the model for future use or deployment in various applications.

## Testing and Results
To validate the model's performance, a separate Jupyter notebook was used for testing the saved model. The accuracy achieved during testing was 85.36%, confirming the model's robustness and effectiveness in facial emotion detection.

## Future Scope
While the current model yields promising results, there are several avenues for further improvement. Consider exploring advanced CNN architectures such as ResNet, VGG, or DenseNet, which may boost accuracy. Additionally, data augmentation techniques and hyperparameter tuning can potentially enhance the model's generalization capabilities. Moreover, you can investigate transfer learning approaches by using pre-trained models on larger facial emotion datasets, which might lead to even better results.

## How to Use
1. Clone the repository to your local machine.
2. Ensure that you have the required dependencies installed (list them if necessary).
3. Open the "train.ipynb" notebook to train the CNN model on the FER-2013 dataset.
4. After training, the model will be saved in the "model" folder.
5. Use the "test.ipynb" notebook to load the saved model and test it on new images or datasets.

## Contributions
Contributions to this project are welcome! If you have ideas for enhancements, bug fixes, or new features, feel free to submit a pull request.

## License
Specify the license under which this project is shared (e.g., MIT License, Apache License, etc.).

## Acknowledgments
Give credit to any resources, tutorials, or code snippets that were helpful in building this project.

---
*Optional: Add any additional details, project screenshots, or demo videos to provide a more comprehensive understanding of the project.*
