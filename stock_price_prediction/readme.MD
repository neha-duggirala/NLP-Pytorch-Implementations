# Stock Price Prediction System Design

## 1. **Understanding the Data**
The dataset used for this project contains historical stock price data with the following features:
- **Open**: Opening price of the stock.
- **High**: Highest price of the stock during the day.
- **Low**: Lowest price of the stock during the day.
- **Close**: Closing price of the stock.
- **Volume**: Number of shares traded.
- **Last**: The target variable representing the final price of the stock.

Columns such as `Date` and `Volume` were dropped as they were deemed irrelevant for the prediction task.

---

## 2. **Data Preparation**
To ensure the data is ready for modeling, the following steps were performed:
- **Scaling**: StandardScaler was used to normalize the data, ensuring all features are on the same scale.
- **Splitting**: The dataset was split into training, validation, and test sets to evaluate model performance on unseen data.
- **Reshaping**: The data was reshaped to meet the input requirements of various models, such as PCA and RNNs.

---

## 3. **Exploratory Data Analysis (EDA)**
EDA was conducted to understand the relationships between features:
- A **correlation matrix** was plotted, revealing high multicollinearity among features.
- This multicollinearity necessitated the use of dimensionality reduction techniques like PCA to simplify the dataset.

---

## 4. **Dimensionality Reduction**
To address multicollinearity, **Principal Component Analysis (PCA)** was applied:
- PCA reduced the dimensionality of the dataset while retaining the most important information.
- This step improved the performance of models sensitive to multicollinearity, such as linear regression.

---

## 5. **Modeling Techniques**
Several models were explored to predict stock prices:

### a. **Linear Regression**
- **Assumptions**: Assumes a linear relationship between features and the target variable, and no multicollinearity.
- **Performance**: Struggled due to high multicollinearity, even after applying PCA.
- **Trade-offs**: Simple and interpretable but unsuitable for non-linear relationships.

### b. **XGBoost**
- **Advantages**: Handles non-linear relationships and is robust to multicollinearity.
- **Performance**: Outperformed linear regression, capturing complex patterns in the data.
- **Trade-offs**: Computationally intensive and requires careful hyperparameter tuning.

### c. **Support Vector Regression (SVR)**
- **Advantages**: Effective for small datasets and can model non-linear relationships using kernels.
- **Performance**: Performed reasonably well but required significant tuning.
- **Trade-offs**: Sensitive to hyperparameters and less scalable to large datasets.

### d. **Recurrent Neural Networks (RNN)**
- **Advantages**: Designed for sequential data, capturing temporal dependencies effectively.
- **Performance**: Delivered strong results, leveraging the temporal nature of stock price data.
- **Trade-offs**: Computationally expensive and requires extensive training and tuning.

---

## 6. **Evaluation Metrics**
The models were evaluated using the following metrics:
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values. Lower values indicate better performance.
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values. Less sensitive to outliers than MSE.
- **R² Score**: Indicates the proportion of variance in the target variable explained by the model. Higher values indicate better performance.

A comparative study of these metrics revealed that XGBoost and RNNs consistently outperformed other models.

---

## 7. **Trade-offs and Final Recommendation**
- **Linear Regression**: Simple and interpretable but unsuitable for this dataset due to multicollinearity and non-linear relationships.
- **XGBoost**: Robust and accurate but computationally intensive.
- **SVR**: Effective for small datasets but less scalable and sensitive to hyperparameters.
- **RNN**: Best suited for sequential data but requires significant computational resources and careful tuning.

### **Best Method**
Based on the evaluation metrics and trade-offs, **RNN (LSTM)** is the best method for this case. It effectively captures the temporal dependencies in the stock price data, leading to more accurate predictions. While computationally expensive, its performance justifies the cost for this application.

---

## 8. **Conclusion**
This study highlights the importance of understanding the dataset, addressing multicollinearity, and selecting appropriate models based on the problem's requirements. RNNs emerged as the most suitable method for stock price prediction due to their ability to model temporal patterns, making them a valuable tool for time-series forecasting tasks.