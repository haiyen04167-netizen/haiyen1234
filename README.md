
## **Project Title**

**Loan Amount Prediction and Analysis using Kiva Dataset**

---

## **1. Overview**

This project analyzes loan data from the Kiva platform to explore trends, correlations, and predict loan amounts using machine learning techniques. The study focuses on understanding how different factors such as funded amount and loan term influence the total loan amount.

---

## **2. Objectives**

* Analyze loan trends over time
* Identify correlations between numerical variables
* Visualize key patterns in the dataset
* Build a prediction model for loan amount
* Evaluate model performance using appropriate metrics

---

## **3. Dataset**

The project uses the following datasets:

* `kiva_loans.csv`
* `kiva_mpi_region_locations.csv`
* `loan_theme_ids.csv`
* `loan_themes_by_region.csv`

These datasets contain information about loan transactions, geographic regions, and loan themes.

---

## **4. Requirements**

Make sure you have the following Python libraries installed:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

You can install them using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## **5. How to Run the Project**

1. Place all dataset files in the same directory as the Python script
2. Open the project folder in your IDE (e.g., VS Code, PyCharm)
3. Run the main file:

```bash
python main.py
```

---

## **6. Project Workflow**

### **Step 1: Data Loading**

The program loads multiple datasets using pandas and displays sample data for verification.

### **Step 2: Data Preprocessing**

* Convert date column to datetime format
* Filter data between January and June 2014

### **Step 3: Data Analysis & Visualization**

* Plot loan trends over time
* Generate correlation heatmap
* Visualize relationships between variables

### **Step 4: Machine Learning Model**

* Model used: Decision Tree Regressor
* Input features:

  * Funded amount
  * Term in months
* Output:

  * Loan amount

### **Step 5: Model Evaluation**

* Mean Squared Error (MSE)
* R² Score

### **Step 6: Visualization of Results**

* Scatter plot (Actual vs Predicted)
* Line chart comparison

---

## **7. Results**

The model predicts loan amounts based on selected features. Performance is evaluated using:

* **MSE**: Measures prediction error
* **R² Score**: Measures model accuracy

---

## **8. Output**

The program produces:

* Loan trend graph
* Correlation heatmap
* Prediction vs actual comparison plots
* Dataframe showing predicted values

---

## **9. Notes**

* Ensure all dataset files are correctly named and located in the same folder
* The model can be improved by adding more features or using advanced algorithms

---

## **10. Author**

* HUFLIT Student Project
* Course: Computational / Data Analysis

