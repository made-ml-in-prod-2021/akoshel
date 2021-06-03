import numpy as np
import pandas as pd
import requests
import time

if __name__ == "__main__":
    data = pd.read_csv("../homework1/data/heart.csv").iloc[:100]
    for i in range(100):
        request_data = data.iloc[i].to_dict()
        request_data["id"] = i
        print(request_data)
        response = requests.get(
            "http://127.0.0.1:8000/predict",
            json=[request_data],
        )
        print(response.status_code)
        print(response.json())
        time.sleep(0.05)