# Arabic-Font-Recognition
An Arabic Font Recognition System: Given an image containing a paragraph written in Arabic, our system classifies the paragraph into one of four fonts, numbered from 0 to 3.
<div align="center">

| Font Code | Font Name              |
|-----------|------------------------|
| 0         | Scheherazade New       |
| 1         | Marhey                 |
| 2         | Lemonada               |
| 3         | IBM Plex Sans Arabic   |

</div>

## Project Pipeline
The workflow begins with loading the image data, followed by a crucial preprocessing step to remove noise and enhance text clarity.
Next, the Scale-Invariant Feature Transform (SIFT) algorithm is used for feature extraction, identifying key features invariant to scale and rotation.
The data is then refined through K-means clustering, which organizes features into distinct groups based on similarity.
Subsequently, machine learning algorithms are employed to train predictive models for font recognition, optimizing performance and accuracy.
Finally, the trained model is validated with test images to ensure its effectiveness and reliability in real-world scenarios.

## Preprocessing Module
In our preprocessing stage, we focused on addressing noise by using median and bilateral filters. The median filter effectively removed salt-and-pepper noise from the image data.

To address blurry images, we used several filters. Canny edge detection helped identify sharp edges in blurred backgrounds, improving clarity, while the unsharp mask filter enhanced image details by boosting local contrast.

By using noise reduction and image enhancement techniques, our preprocessing stage effectively tackled common challenges in real-world images. This approach improved image quality and provided a strong foundation for subsequent analyses, leading to more reliable results in later stages.

## Feature Selection/Extraction
We chose SIFT due to its robustness in handling various challenges, such as differences in scale and rotation.
This resilience makes SIFT particularly well-suited for our needs, ensuring accurate and consistent image extraction across diverse scenarios.

## Model Selection
In our detailed model selection process, we examined various methodologies, including hybrid approaches and standalone models like Random Forests, KNN, and SVM.
Each model was rigorously evaluated for computational efficiency, predictive accuracy, and scalability. Ultimately, we chose Random Forest due to its superior accuracy.

## Performance Analysis Module
In our performance analysis, we first partitioned the data into training and testing sets with an 80:20 ratio. We then trained our models on the training data and evaluated their predictions using the test data. You can access the dataset used for this analysis [here](https://www.kaggle.com/datasets/breathemath/fonts-dataset-cmp).
|        Model      |   Feature Extraction Method |        Accuracy        |
|-------------------|-----------------------------|------------------------|
| SVM               |  Histogram                  | 78.38%                 |
| SVM               |  SIFT                       | 87.88%                 |
| KNN               |  SIFT                       | 98%                    |
| Random Forest     |  SIFT                       | 98.5%                  |

## Deployment
To streamline the testing process, we deployed our module on Azure and made it accessible through a public API using Flask. This deployment allowed for convenient and efficient testing of our models.

## ©️Developers

| Name                 |         Email          |
|----------------------|:----------------------:|
| Fares Atef           | faresatef553@gmail.com |
| Ghaith Mohamed       |  gaoia123@gmail.com    |
| Amr ElSheshtawy      | Sheshtawy321@gmail.com |
| Amr Magdy            |  amr4121999@gmail.com  |
