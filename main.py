import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
sns.set()
plt.style.use('seaborn-v0_8-whitegrid')
kiva_loans = pd.read_csv("kiva_loans.csv")
mpi_location = pd.read_csv(r"C:\final chapter\kiva_mpi_region_locations.csv")
loan_theme_ids = pd.read_csv("loan_theme_ids.csv")
loan_themes_region = pd.read_csv("loan_themes_by_region.csv")
print(kiva_loans.head())
print(mpi_location.head())
print(loan_theme_ids.head())
print(loan_themes_region.head())
# dùng dataset đã load
kiva_loans["date"] = pd.to_datetime(kiva_loans["date"])
filtered_data = kiva_loans[
    (kiva_loans["date"] >= "2014-01-01") &
    (kiva_loans["date"] <= "2014-06-30")

]
# group theo ngày
loan_trend = filtered_data.groupby("date")["loan_amount"].sum()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(loan_trend)

plt.title("Loan Amount (Jan - Jun 2014)")
plt.xlabel("Date")
plt.ylabel("Loan Amount")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
numeric_data = kiva_loans.select_dtypes(include=['number'])

print(numeric_data.corr())
plt.figure(figsize=(10, 6))

sns.heatmap(
    numeric_data.corr(),
    annot=True,        # hiện số
    fmt=".2f",         # 2 chữ số thập phân
    cmap="coolwarm"    # màu đẹp
)

plt.title("Correlation Heatmap")
plt.show()
import numpy as np

x = kiva_loans[["funded_amount", "term_in_months"]]
y = kiva_loans["loan_amount"]

x = x.to_numpy()
y = y.to_numpy()

y = y.reshape(-1, 1)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# chia dữ liệu
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# sửa shape y
ytrain = ytrain.ravel()
ytest = ytest.ravel()

# tạo model
model = DecisionTreeRegressor(random_state=42)

# train
model.fit(xtrain, ytrain)

# predict
ypred = model.predict(xtest)
mse = mean_squared_error(ytest, ypred)
r2 = r2_score(ytest, ypred)

print("MSE:", mse)
print("R2 Score:", r2)
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(ytest, ypred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
data = pd.DataFrame(data={"Predicted Rate": ypred.flatten()})
print(data.head())
result = pd.DataFrame({
    "Actual": ytest.flatten(),
    "Predicted": ypred.flatten()
})

print(result.head())
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(result["Actual"].values, label="Actual")
plt.plot(result["Predicted"].values, label="Predicted")

plt.legend()
plt.title("Actual vs Predicted")
plt.show()