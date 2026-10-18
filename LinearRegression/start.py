import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

MILEAGE = data.mileage
PRICES = data.price


act_y = data.price
x = (data.mileage - data.mileage.mean()) / data.mileage.std()

m = 1
b = 1
lr = 0.1
N = len(data)
epocs = 250


for i in range(250):

    pred_y = m * x + b

    # mse = - 1/len(data) * sum((pred_y - act_y) ** 2)
    dm = - 2/N * sum(x * (act_y - pred_y))
    db = -2/N * sum(act_y - pred_y)

    m = m - lr * dm
    b = b - lr * db

    if i % 50 == 0:
        print(i)


plt.scatter(x, act_y, color="blue", label="Actual Data")
plt.plot(x, m * x + b, color="red", label="Fitted Line")  # Draws the line

plt.xlabel("Scaled Mileage")
plt.ylabel("Price")
plt.legend()    
plt.show()
