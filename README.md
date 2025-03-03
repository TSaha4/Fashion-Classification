# Fashion-Classification

##  Overview
This project classifies images from the **Fashion MNIST** dataset using **Logistic Regression**. The dataset contains grayscale images (28x28 pixels) of different clothing items. Our goal is to build a simple machine learning model that can predict the category of an image.

##  Technologies Used
- **Python**
- **NumPy & Pandas** (Data handling)
- **Matplotlib & Seaborn** (Data Visualization)
- **Scikit-learn** (Machine Learning)

##  Dataset
- The dataset contains **60,000 training** and **10,000 test** images.
- Each image is **28x28 pixels** and belongs to one of **10 categories**:
  1. T-shirt/top
  2. Trouser
  3. Pullover
  4. Dress
  5. Coat
  6. Sandal
  7. Shirt
  8. Sneaker
  9. Bag
  10. Ankle boot

##  Steps Performed

### 1️ Importing Libraries
- Import essential Python libraries like **NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn**.
- These libraries help with **data handling, visualization, and machine learning**.

### 2️ Loading the Dataset
- Load the dataset from a **CSV file** containing pixel values and corresponding labels.
- Use Pandas to **read and inspect** the dataset.

### 3️ Handling Missing Values
- Check for **missing values** in the dataset.
- Drop any rows with missing values to maintain data consistency.
- Fill any remaining missing values with **zero** to avoid errors.

### 4️ Preparing Data
- Separate the **labels (y)** and **pixel values (X)**.
- Reshape the pixel values into **28x28 images** for better visualization.
- Normalize the pixel values by **scaling them between 0 and 1** to improve model performance.

### 5️ Visualizing Sample Images
- Select sample images from the dataset.
- Plot them using **Matplotlib** to understand the distribution of different clothing items.
- Assign appropriate **titles** to each image for better clarity.

### 6️ Splitting Data into Train & Test Sets
- Divide the dataset into **training (80%)** and **testing (20%)** sets.
- Ensure that the split is **stratified** to maintain class distribution.
- Use `train_test_split()` from Scikit-learn for an efficient split.

### 7️ Training Logistic Regression Model
- Initialize a **Logistic Regression** model using Scikit-learn.
- Use the **multinomial approach** since there are multiple classes.
- Set **max iterations to 1000** for proper convergence.
- Train the model using the **training dataset**.

### 8️ Evaluating the Model
- Predict the labels of the **test dataset** using the trained model.
- Compute the **accuracy score** to measure performance.
- Display the classification **report** for detailed metrics like precision, recall, and F1-score.

### 9️ Confusion Matrix
- Generate a **confusion matrix** to analyze classification errors.
- Use Seaborn to **visualize** the confusion matrix.
- Identify which categories the model struggles with the most.

##  Results
- The model achieved a **reasonable accuracy** of **85.26%** on the test dataset.
- The confusion matrix provided insights into **misclassified categories**.

