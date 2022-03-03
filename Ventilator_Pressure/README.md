# Ventilator_Pressure_Prediction

![image](https://user-images.githubusercontent.com/70372577/136055565-9bc765c7-9ade-4d73-9c4f-4468028e44c3.png)

you can download the data at https://www.kaggle.com/c/ventilator-pressure-prediction/data


# hitmap
![aas](https://user-images.githubusercontent.com/70372577/138805142-190a8c1e-2aca-40cf-87b0-d4e717af5bc5.png)
색갈별로 상관계수를 나타냄 u_out, u_in, time_step이 prediction_value인 pressure과 관련이 깊음을 알 수 있으므로,
더 나은 정확도를 위해 이 부분의 feature engineering을 주의깊게 해야함 

# Cross Validation Loss(Mean Average Error) : 0.69 달성!!!
![image](https://user-images.githubusercontent.com/70372577/139282017-f500b9aa-e0ba-4421-8845-2b5d96ae8535.png)

MAE Loss 기준
CV : 0.69
LB : 1.1
