# Telecom Customer Churn Prediction
![ROC](https://github.com/shirinshaik/Machine-Learning_classification-Customer-Churn-Rate_Project/assets/113626760/21c80a30-6692-476e-94bd-3b80bdddb911)

## Objective
The main objective of this project is to develop a predictive model that can identify potential customer churn in a subscription-based business. Customer churn is a critical metric for any subscription-based service, as it directly impacts revenue and growth. By accurately predicting churn, businesses can take proactive measures to retain valuable customers.

## Introduction
Customer churn, also known as customer attrition, refers to the rate at which customers stop doing business with a company over a given period of time. It is a key concern for businesses in subscription-based industries such as SaaS, telecommunications, and media streaming. Understanding and predicting churn can help companies implement strategies to retain customers and sustain growth.

This project leverages machine learning techniques to analyze historical customer data and build a predictive model. The model aims to classify customers into churn or non-churn categories based on features such as usage patterns, customer demographics, and customer service interactions.

## Features
- Understand the Problem Statement and Business Case
- Exploratory Data Analysis
- Data Visualization
- Identification of Feature Importance and Preparation of Data Before Model Training
- Train and Evaluate a Logistic Regression Classifier
- Train and Evaluate a Support Vector Machine Classifier
- Train and Evaluate a Random Forest Classifier
- Train and Evaluate a K-Nearest Neighbour (KNN) Classifier
- Train and Evaluate a Naïve Bayes Classifier
- Plot ROC Curves For The 5 Models and Find AUC Scores

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Usage
The project is divided into tasks, each focusing on a specific classifier to develop a predictive model. Here's a brief overview of the tasks:

### TASK 1: UNDERSTAND THE PROBLEM STATEMENT AND BUSINESS CASE
In this hands-on project, we will train several classification algorithms namely Logistic Regression, Support Vector Machine, K-Nearest Neighbors, and Random Forest Classifier to predict the churn rate of Telecommunication Customers.

• Telecom service providers use customer attrition analysis as one of their key business metrics because the cost of retaining an existing customer is far less than acquiring a new one. 

• Machine Learning algorithms help companies analyze customer attrition rate based on several factors which includes various services subscribed by the customers, tenure rate, gender, senior citizen, payment method, etc.

### TASK 2: IMPORT LIBRARIES/DATASETS AND PERFORM EXPLORATORY DATA ANALYSIS

In this task, we lay the foundation for our exploratory data analysis (EDA) by setting up a Jupyter notebook environment and importing essential Python libraries. These libraries are pivotal for various aspects of data analysis.

- NumPy (np): This library enables efficient handling of multi-dimensional arrays and numerical computations, forming the backbone of numerical operations.
- Pandas (pd): Pandas provides a robust framework for data manipulation, offering powerful data structures like DataFrames for seamless handling of tabular data.
- Matplotlib.pyplot (plt): As part of the comprehensive data visualization library, Matplotlib, pyplot equips us with a user-friendly interface for generating static visualizations.
- Seaborn (sns): This high-level data visualization library focuses on creating aesthetically pleasing and informative statistical graphics, adding depth to our visualizations.
- Plotly Express (px): Offering an accessible means to create interactive plots, Plotly Express extends our visualization capabilities.
- Plotly Offline and Cufflinks (download_plotlyjs, init_notebook_mode, plot, iplot, cf): These components of Plotly enhance offline functionality and facilitate seamless integration with Pandas for streamlined plotting.
- Jupyterthemes and jtplot: These tools allow us to customize the appearance of our Jupyter notebooks, providing a tailored environment for our analysis.

By importing these libraries, we ensure that we have the necessary tools at our disposal for subsequent phases of the exploratory data analysis process. This includes tasks such as data manipulation, visualization, and customization of the Jupyter notebook interface to suit our analytical needs. This step lays the groundwork for a comprehensive and effective data exploration.

### Task 3: PERFORM DATA VISUALIZATION
In this task, we leverage various visualization techniques to gain deeper insights into the telecom dataset and its underlying patterns. The visualizations are generated using Python libraries such as Matplotlib, Seaborn, and Plotly.
- Histograms: We start by creating histograms using Matplotlib to visualize the distributions of different features in the dataset. This allows us to gain an understanding of the spread and frequency of values for each variable.
- Pie Chart: Using Plotly, we construct a pie chart that provides a clear breakdown of the percentage of retained (0) and churned (1) customers. This visualization offers a concise overview of customer retention.
- Categorical Histograms: We utilize Plotly Express to generate histograms that illustrate the distribution of telecom customers based on their usage of international plan services, categorized by churned or retained status.
- Correlation Matrix Heatmap: A correlation matrix is created using Seaborn to visualize the relationships between different variables. This heatmap provides insights into how various features are correlated with each other, helping identify potential patterns.
- Kernel Density Estimation (KDE) Plots: These plots are employed to compare the density distributions of features like total day charges and total evening charges for both retained and churned customers. The KDE plots facilitate a visual understanding of how charges are distributed among different customer groups.
- Voice Mail Plan vs. Churn Plot: Using Plotly, we generate a histogram to explore the correlation between the usage of voice mail plans and customer churn. This visualization provides valuable insights into the impact of voice mail plan usage on customer retention.

By employing these visualization techniques, we enhance our understanding of the dataset, uncovering meaningful relationships and patterns that contribute to a more comprehensive exploratory data analysis. These visualizations serve as crucial tools for making informed decisions in subsequent stages of the project.

### Task 4: IDENTIFY FEATURE IMPORTANCE & PREPARE THE DATA BEFORE MODEL TRAINING
In this step, we focus on identifying feature importance and preparing the data for model training. This is a critical stage in building a machine learning model, as unnecessary features can negatively impact training speed, model interpretability, and its ability to generalize to new data.

First, we begin by considering the dataset 'telecom_df'. We drop certain features, specifically "class", "area_code", and "phone_number", as they are deemed irrelevant for training the model. The "class" feature serves as our target variable, so it is separated and assigned to 'y'. Meanwhile, the remaining features, excluding the aforementioned ones, are designated as input features and assigned to 'X'.

Next, to evaluate the model's performance, we perform a train/test split on the data. This involves dividing the dataset into two sets: one for training the model ('X_train' and 'y_train') and the other for testing its performance ('X_test' and 'y_test'). In this case, we allocate 80% of the data for training and 20% for testing.

Finally, a verification step is recommended to ensure that the train/test split was executed successfully. This helps confirm that the data has been appropriately divided, enabling us to proceed with training and evaluating the model.

### Task 5: TRAIN AND EVALUATE A LOGISTIC REGRESSION CLASSIFIER
In this step, we employ a logistic regression classifier to build a predictive model. Logistic regression is a widely used algorithm for binary classification tasks. We start by importing necessary modules including LogisticRegression for model creation and classification_report along with confusion_matrix for performance evaluation.

After initializing and training the logistic regression model on the training data, we make predictions on the test set. The results are stored in y_predict. Subsequently, we generate a detailed classification report which provides key metrics on the model's performance.

From the report, we note that the accuracy stands at 82%. Precision values for 'class 0' and 'class 1' are 87% and 45% respectively, indicating the proportion of correctly predicted instances for each class. However, recall for 'class 0' is high at 99%, while for 'class 1' it's notably lower at 7%. This suggests that the model may struggle with identifying 'class 1' instances. Further, the F1 scores balance precision and recall, yielding values of 93% for 'class 0' and 12% for 'class 1'.

Examining the confusion matrix, we find that 120 samples have been misclassified. This substantial number may contribute to the lower accuracy, precision, and recall scores observed.

### Task 6: TRAIN AND EVALUATE A SUPPORT VECTOR MACHINE CLASSIFIER
In this step, we implement a Support Vector Machine (SVM) classifier for binary classification. First, we import necessary modules including LinearSVC for SVM classification and CalibratedClassifierCV to calibrate the classifier for probability score output.

The SVM model is trained on the training data and subsequently used to make predictions on the test set. The classification report provides a detailed summary of the model's performance, including precision, recall, and F1-scores for both 'class 0' and 'class 1'.

Upon evaluation, we observe that the classifier has achieved suboptimal performance, particularly for 'class 1' where the recall is quite low at 20%. The overall accuracy is 88%.

Additionally, the confusion matrix illustrates that 9 samples in 'class 0' and 100 samples in 'class 1' have been misclassified. By employing CalibratedClassifierCV, we marginally improve the classifier's performance, resulting in an increase in correctly classified data points from '9' to '26' in 'class 1'.

### Task 7: TRAIN AND EVALUATE A RANDOM FOREST CLASSIFIER
In this step, we employ a Random Forest Classifier for binary classification. The model is trained on the training data and then tested on the validation set. The resulting classification report provides a detailed breakdown of precision, recall, and F1-scores for both 'class 0' and 'class 1'.

The evaluation demonstrates that the Random Forest Classifier outperforms previous models. It achieves an impressive accuracy of 97%, with high precision for both 'class 0' (97%) and 'class 1' (95%). Recall rates are also commendable, particularly for 'class 0' (99%), although slightly lower for 'class 1' (78%). The F1 scores indicate a robust balance between precision and recall.

Upon examining the confusion matrix, we observe that only 5 samples in 'class 0' and 29 samples in 'class 1' have been misclassified. This showcases the substantial improvement in performance compared to earlier classifiers.

### Task 8: TRAIN AND EVALUATE A K-NEAREST NEIGHBOUR (KNN)
In this step, a K-Nearest Neighbour (KNN) Classifier is trained on the data and assessed on the validation set. The classification report provides detailed metrics including precision, recall, and F1-scores for both 'class 0' and 'class 1'.

The KNN model exhibits respectable performance with an accuracy of 89%. Precision for 'class 0' and 'class 1' is 90% and 68% respectively. However, recall rates are noticeably lower, particularly for 'class 1' at 32%. The F1 scores indicate a reasonable balance between precision and recall, but it is still outperformed by the Random Forest Classifier.

In the confusion matrix, we note that 91 samples in 'class 1' and 20 samples in 'class 0' have been misclassified. While KNN demonstrates improved performance compared to SVM, it still falls short of the Random Forest Classifier.

### Task 9: TRAIN AND EVALUATE A NAIVE BAYES CLASSIFIER
In this step, a Naive Bayes Classifier, specifically the Gaussian Naive Bayes model, is trained and assessed on the validation set. The classification report provides detailed metrics including precision, recall, and F1-scores for both 'class 0' and 'class 1'.

The Gaussian Naive Bayes model demonstrates an accuracy of 88%, indicating a reasonable performance. However, it falls slightly short of the Random Forest Classifier. Precision for 'class 0' and 'class 1' is 94% and 53% respectively, while recall rates are 92% and 59% respectively. The F1 scores reflect a relatively balanced trade-off between precision and recall.

The confusion matrix shows that 71 samples in 'class 0' and 54 samples in 'class 1' have been misclassified. While the Gaussian Naive Bayes model offers a respectable performance, it is not as robust as the Random Forest Classifier.

### Task 10: PLOT ROC CURVES FOR THE 5 MODELS AND FIND AUC SCORES
In this task, ROC curves and AUC scores were generated for five different models: Logistic Regression, Support Vector Machine (SVM), Random Forest, K-Nearest Neighbors (KNN), and Naive Bayes. The ROC curves provide a visual representation of the model's performance, illustrating the trade-off between true positive rate (sensitivity) and false positive rate.

The AUC scores quantify the model's ability to distinguish between the two classes. Among the models, Random Forest achieved the highest AUC score of 0.92, indicating superior performance in classifying churned and retained telecom customers. The worst performer was K-Nearest Neighbors with an AUC score of 0.70.

The ROC curve plot visually reinforces the Random Forest model's effectiveness, as it exhibits the highest curve, followed by SVM. This graphically confirms that Random Forest outperformed the other models in terms of classifying customers.

Overall, this analysis confirms the Random Forest model as the most reliable classifier for this particular dataset, reinforcing earlier findings in the classification reports and confusion matrices.

## Notable Insights
- Features like "total_day_minutes" and "total_intl_minutes" were identified as significant predictors of churn.
- The Random Forest Classifier outperformed other models, achieving an accuracy of 97%.
- Random Forest demonstrated the highest AUC score of 0.92 in ROC curve analysis, signifying its superior ability to distinguish between churned and retained customers.
- Despite high accuracy, the Random Forest model misclassified a small number of samples, particularly 5 in 'class 0' (retained) and 29 in 'class 1' (churned).

## Conclusion
In conclusion, the exploratory data analysis revealed crucial insights into the telecom customer dataset. Notably, the Random Forest Classifier emerged as the most effective model for predicting customer churn, achieving an impressive accuracy of 97%. This classifier demonstrated superior performance in identifying features critical for classification. Despite its high accuracy, the model did misclassify a small number of samples. The dataset showcased class imbalance, potentially influencing model performance. Moreover, the Random Forest model exhibited the highest AUC score, indicating excellent discriminative ability. This comprehensive analysis underscores the significance of employing Random Forest in customer churn prediction for this particular dataset.

★ Fine-tuning hyperparameters and exploring more advanced ensemble methods, such as Gradient Boosting or XGBoost, could potentially enhance model performance even further.
★ Predicting customer churn is crucial for telecom companies to implement targeted retention strategies. The insights derived from this analysis can inform decisions on customer engagement, service offerings, and marketing initiatives.
