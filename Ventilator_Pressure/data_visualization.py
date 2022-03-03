import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_path = "./input/train.csv"
test_path = "./input/test.csv"
sample_sub = "./input/sample_submission.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path) 

# print(train_data.shape)

# # Checking for missing values & size of the data
# print(f"Rows in training data : {train_data.shape[0]}")
# print(f"Rows in test data : {test_data.shape[0]}")
# print(f"Columns in train_data : {train_data.columns.tolist()}")
# print("Target column: pressure\n")

# print(f"Missing values in train data\n{train_data.isna().sum().to_frame()}\n")
# print(f"Missing values in test data\n{test_data.isna().sum().to_frame()}\n")

# ventilation_cycle = train_data[train_data['breath_id']==3]
# print(f"Unique value counts in each time stamp\n{ventilation_cycle.nunique()}\n")

def draw_1_cycle(ventilation_cycle):
    v_id = ventilation_cycle[ventilation_cycle.u_out==1].id.values[0]
    # u_out이 1인것 중 첫번쨰 id의 값을 가져옴
    plt.figure(figsize=(18, 5))
    
    for col in ventilation_cycle.columns:
        if col=="id":
            continue
        plt.plot(ventilation_cycle['id'], ventilation_cycle[col], label=col)
    
    l = ventilation_cycle.max().values
    l.sort()
    plt.vlines(x = v_id, ymin = 0.1, ymax = l[-2], linestyles="dotted", color="grey")
    plt.legend(loc = 'best')
    plt.title("Visualization of one ventilation cycle(~3s)")
    plt.show()

for i in range(1, 20, 4):
    draw_1_cycle(train_data[train_data['breath_id']==i])