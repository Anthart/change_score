import pandas as pd
import json
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

methods = {
    "first_method": {
        "very easy": 0.0, 
        "easy": 0.125, 
        "neutral": 0.375, 
        "difficult": 0.625, 
        "very difficult": 0.875
    },
    "second_method": {
        "very easy": 0.0, 
        "easy": 0.18969275570670296, 
        "neutral": 0.35131657758634227, 
        "difficult": 0.5880268389141909, 
        "very difficult": 0.8111511532380951
    }
} 

selected_method = "first_method"

route_files = "files/"
route_results = "results/"
file_name = "resultado_strat2_version11_Test"

try:
    df = pd.read_excel(route_files + file_name + '.xlsx', index_col=0)

    df["change"] = df["Respuesta GPT3"].apply(
        lambda row,value: value[row], 
        args =(methods[selected_method], )
        ) 

    true = df.loc[:, "complexity"]
    predicted = df.loc[:, "change"]

    metrics = {
        "MAE": round(mean_absolute_error(true, predicted), 4),
        "MSE": round(mean_squared_error(true, predicted), 4),
        "RMSE": round(mean_squared_error(true, predicted, squared=False), 4),
        "R2": round(r2_score(true, predicted), 4),
        "Pearson": round(true.corr(predicted, method='pearson'), 4),
        "Spearman": round(true.corr(predicted, method='spearman'), 4)
    }

    print(metrics)

    tf = open(route_results + file_name.split("_")[2] + "_" + selected_method + ".json", "w")
    json.dump(metrics,tf)
    tf.close()
except FileNotFoundError as err:
    print(err)
except Exception as err:
    print(err)