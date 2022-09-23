# Neural Network Diagnosing Diabetes

Training a [TensorFlow]-based fully connected neural network to diagnose patients with diabetes.

---

## Problem Overview

The [diabetes dataset] has been used by the machine learning community to train classifiers to detect diabetes. It has 768 instances with 9 features, one of them being the response variable that tells if a person has diabetes or not. The dataset is built with data referring to females, older than 21 and of Pima Indian heritage. Feature names are self-explanatory to some extent; the table below has a description for each feature:

| Feature | Description |
|:------:|:---------------------:|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration a 2 hours in an oral glucose tolerance test |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index (weight in kg/(height in m)^2) |
| DiabetesPedigreeFunction|  Likelihood of diabetes based on family history |
| Age | Age in years |
| Outcome | Class variable: 0 if diabetes is not present; 1 if diabetes is present |

The objective is to create a classifier to predict if a person has diabetes or not, given their data in the above form.

## Analysis Introduction

In this project, we show how to preprocess and select the most relevant variables to train a simple fully connected neural network. Explanatory variables are subject to being discarded due to high correlation with other explanatory variables (for carrying similar information),  or due to low correlation with the response variable (for not making good contributions to the model). Correlations are evaluated with pair plots and a correlation matrix:

![corr_matrix_diabetes_nn](https://user-images.githubusercontent.com/33037020/191629012-4878719c-1ba4-40a9-8723-b5ccf738f094.png)

With the 5 most meaningful explanatory variables selected from the 8 original, a neural network with two hidden layers is trained. Two hidden layers have been known in the [literature] for being capable of modeling most practical problems. Since our diabetes dataset already presents features in a numerical manner, it is not necessary to add more layers to perform feature extraction.

Results are promising given the amount of available data (just 768 instances), staying around 82% in accuracy. Our trained fully connected neural network achieves an F1-score of 0.87 considering class 0 only (not having diabetes). Only 268 instances are available for class 1 (having diabetes), and 30% of them are separated for validation and testing. Consequently, the model is not able to achieve optimal performance considering class 1 only, with F1-score staying at 0.71. Data augmentation would probably improve the performance of the model.

[//]: #

[diabetes dataset]: <https://www.kaggle.com/datasets/mathchi/diabetes-data-set>
[literature]: <http://www.faqs.org/faqs/ai-faq/neural-nets/part3/section-9.html>
[TensorFlow]: <https://www.tensorflow.org>
