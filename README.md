# Cervical Cancer Model

This project uses several Machine Learning models, including Logistic Regression (LR), Support Vector Machines (SVM), Partial Least Squares Regression (PLSR), and Random Forest (RF) to predict risk of cervical cancer using the [Cervical Cancer Risk Factors](https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors) dataset from the UCI Machine Learning repository.

Ultimately, even with normalization of the data and attempts to remediate missing values, we found poor prediction of cervical cancer risk upon cross-validation with a high false negative rate. Looking at PLSR for example, there was poor separation between negative samples and positive samples in the scores plot.

We were not able to replicate other papers which claimed to accurately predict cervical cancer in this dataset using decision tree/jungle approaches.

## Challenges

Due to the sensitive nature of the collected data, it would be reasonable for patients to omit data in a non-random way and/or misreport their own data, to the detriment of the model.

## Next steps

Future attempts to predict cervical cancer using this data set could use deep learning methods like neural networks.
