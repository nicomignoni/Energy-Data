# Energy Forecasting with NeuralProphet, Darts and MongoDB
## Context and Objectives

<img src="img/smart_grid_data.png" alt="Drawing" style="width: 600px;" align="right"/>

Anticipating energy generation and load request can improve the efficiency of the energy market, since it makes it easier to match the demand with the offer. One of the most common problems in the energy scenario is predicting the amount of generated and requested power over time. 
This is particularly important when dealing with renewable energy sources, such as photovoltaic and eolic, which are heavily charachterized by intermittent production. 


The objective of the present project work is to create a machine-learning prototype pipeline for forecasting solar and eolic generation. There are several requirements that an efficient system should meet, which are not only related to the precision of the model. Since energy production and demand data are constantly generated, an important parameter to consider is the speed of the training phase. In fact, training on a large dataset once is unfeasible, since the environmental conditions affecting these fenomena are constantly changing.

## Tools and Data
In this project we'll focus on two machine-learning based tools, i.e, [NeuralProphet](https://neuralprophet.com/html/index.html) (Triebe *et al.*, 2021) and [Darts](https://unit8co.github.io/darts/index.html) (Herzen *et al.*, 2021). For our purposes, we need two different forecasting tools since Neural Prophet cannot deal with multivariate data, but, differently from Darts, it provides a hybrid model which bridges the gap between interpretable classical methods and scalable deep learning models. 

Data consists of a CSV, provided by [Kaggle](https://www.kaggle.com/nicholasjhana/energy-consumption-generation-prices-and-weather?select=weather_features.csv), which is described as follows
>This dataset contains 4 years of electrical consumption, generation, pricing, and weather data for Spain. Consumption and generation data was retrieved from ENTSOE a public portal for Transmission Service Operator (TSO) data. Settlement prices were obtained from the Spanish TSO Red Electric España.

It comprises several energy sources, some of them renewables while other ones thermal. For our purposes, we will focus only on the solar and wind generation data, with the support of the weather data as a past covariate for eolic generation forecasting. Data has been managed using [MongoDB](https://www.mongodb.com/) (Banker *et al.*, 2016), which is easily integrated with Python and pandas.

## Univariate Forecasting 
<img src="img/neural_prophet.png" alt="Drawing" style="width: 250px; float: right;"/>


NeuralProphet is the successor to Facebook Prophet. Both projects have been developed out of the need to bridge the gap between interpretable classical methods and scalable deep learning models. However, Prophet lacks local context, which is essential for forecasting the near term future and is challenging to extend due to its Stan backend. NeuralProphet is a hybrid forecasting framework based on PyTorch and trained with standard deep learning methods. Local context is introduced with auto-regression and covariate modules, which can be configured as classical linear regression or as Neural Networks [1]. 


```python
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pymongo import MongoClient
from imp import reload

from neuralprophet import NeuralProphet, set_random_seed 

# Settings
set_random_seed(42)
cm = 1/2.54
# reload(plt)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans",
    "font.sans-serif": ["Computer Modern Roman"]})
```

### Fetching and processing data
<img src="img/mongo.png" alt="Drawing" style="width: 200px; float: right;">

Data is managed using MongoDB, which is well integrated with `pandas`, allowing for an easy manipulation. MongoDB also provides a Python library from which we can call the `MongoClient` for interacting with the collections. 
Energy data needs some processing, given that there are missing data. For this case, we filled the missing data using the *forward fill* method. NeuralProphet needs the timestamp column to be called `ds`, while the target one `y`, so we will proceed to rename them. 


```python
# Read data, remove NaNs and fix columns names
client = MongoClient(port=27017)
energy_data = client.Energy.HourlyGeneration.find()
generation = pd.DataFrame.from_records(energy_data, columns=["time", "generation solar", "generation wind onshore"])

generation.fillna(method="ffill", inplace=True)
#generation.rename(columns={"time": "ds", "generation solar": "y"}, inplace=True)

generation
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>generation solar</th>
      <th>generation wind onshore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-12-31 23:00:00</td>
      <td>49.0</td>
      <td>6378.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-01-01 00:00:00</td>
      <td>50.0</td>
      <td>5890.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-01-01 01:00:00</td>
      <td>50.0</td>
      <td>5461.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-01-01 02:00:00</td>
      <td>50.0</td>
      <td>5238.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-01-01 03:00:00</td>
      <td>42.0</td>
      <td>4935.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>35059</th>
      <td>2018-12-31 18:00:00</td>
      <td>85.0</td>
      <td>3113.0</td>
    </tr>
    <tr>
      <th>35060</th>
      <td>2018-12-31 19:00:00</td>
      <td>33.0</td>
      <td>3288.0</td>
    </tr>
    <tr>
      <th>35061</th>
      <td>2018-12-31 20:00:00</td>
      <td>31.0</td>
      <td>3503.0</td>
    </tr>
    <tr>
      <th>35062</th>
      <td>2018-12-31 21:00:00</td>
      <td>31.0</td>
      <td>3586.0</td>
    </tr>
    <tr>
      <th>35063</th>
      <td>2018-12-31 22:00:00</td>
      <td>31.0</td>
      <td>3651.0</td>
    </tr>
  </tbody>
</table>
<p>35064 rows × 3 columns</p>
</div>



 Also, NeuralProphet needs the datastamps to be `datetime` with no UCT, so we convert the `ds` column to the correct type.


```python
# Change dates' datatype as "datetime" (with no UCT)
generation['time'] = pd.to_datetime(generation['time'], utc=True).dt.tz_localize(None)

generation.set_index(keys=["time"], inplace=True)
```

Data has a span of 4 years, but we will focus only on the first 6 months of 2017. In fact, sometimes a larger time series affects the performance of the forecasting model negatively, other than adding more computational burden. 


```python
# Solar generation in 2017
interval_start = "2017-01-01 00:00:00"
interval_end   = "2017-07-01 00:00:00"
generation_2017 = generation[interval_start:interval_end]

data = {
    "solar": {"title": "$2017$ Solar Generation", 
              "data": generation_2017[["generation solar"]],
              "target": "generation solar"},
    "wind": {"title": "$2017$ Wind Generation",
             "data": generation_2017[["generation wind onshore"]],
             "target": "generation wind onshore"}
}

for value in data.values(): 
    fig_2017_generation = plt.figure(figsize=(40*cm, 9*cm))
    plt.grid(True, alpha=0.25)
    plt.plot(value["data"], color="b")
    plt.title(value["title"])
    plt.xlabel("Time")
    plt.ylabel("Power generation (kW)")
    plt.xticks(rotation=45)
    plt.show()
```


    
![png](TSDB%202022%20-%20ProjectWork_files/TSDB%202022%20-%20ProjectWork_10_0.png)
    



    
![png](TSDB%202022%20-%20ProjectWork_files/TSDB%202022%20-%20ProjectWork_10_1.png)
    


### Apply log-transformation to data 
It is well-known that log-transformation improves the overall results of forecasting algorithms, given that, in time series analysis, this transformation stabilizes the variance of a series (Lütkepohl *et al.*, 2012).


```python
# log Solar generation in 2017
data["solar"]["proc_data"] = data["solar"]["data"][["generation solar"]].apply(np.log)
data["wind"]["proc_data"] = data["wind"]["data"][["generation wind onshore"]].apply(np.log)

for value in data.values(): 
    fig_2017_generation = plt.figure(figsize=(40*cm, 9*cm))
    plt.grid(True, alpha=0.25)
    plt.plot(value["proc_data"], color="b")
    plt.title(f"{value['title']} (log)")
    plt.xlabel("Time")
    plt.ylabel("(log) Power generation")
    plt.xticks(rotation=45)
    plt.show()
```


    
![png](TSDB%202022%20-%20ProjectWork_files/TSDB%202022%20-%20ProjectWork_12_0.png)
    



    
![png](TSDB%202022%20-%20ProjectWork_files/TSDB%202022%20-%20ProjectWork_12_1.png)
    


### Training the model and forecasting
We train the model with a forecasting window of 24 hours, i.e., the day-ahead window. Tuning the hyperparameters is tricky: `n_lags` determines how far into the past the auto-regressive dependencies should be considered, while the `weekly_seasonality` makes the model look for seasonalities on a weekly basis. The validation set corresponds to $30\%$ of the six-month period.


```python
# Training and Forecast Solar Generation (24 hours)
solar_model = NeuralProphet(n_forecasts=24,
                            n_lags=48,
                            weekly_seasonality=True)

data["solar"]["proc_data"].reset_index(level="time", inplace=True)
data["solar"]["proc_data"].rename(columns={"time": "ds", data["solar"]["target"]: "y"}, inplace=True)

train, test = solar_model.split_df(data["solar"]["proc_data"], valid_p=0.3, freq="h")
metrics = solar_model.fit(train, validation_df=test,  freq="h", plot_live_loss=True)

data["solar"]["model"] = solar_model

metrics.tail(1)
```


    
![png](TSDB%202022%20-%20ProjectWork_files/TSDB%202022%20-%20ProjectWork_14_0.png)
    


    Epoch[137/137]: 100%|█| 137/137 [00:50<00:00,  2.70it/s, SmoothL1Loss=0.00648, MAE=0.514, RMSE=0.697, RegLoss=0, MAE_va
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SmoothL1Loss</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>RegLoss</th>
      <th>SmoothL1Loss_val</th>
      <th>MAE_val</th>
      <th>RMSE_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>136</th>
      <td>0.00648</td>
      <td>0.513772</td>
      <td>0.697065</td>
      <td>0.0</td>
      <td>0.007879</td>
      <td>0.561398</td>
      <td>0.770194</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Training and Forecast Wind Generation (24 hours)
wind_model = NeuralProphet(n_forecasts=24,
                           n_lags=48,
                           num_hidden_layers=2,
                           weekly_seasonality=True)

data["wind"]["proc_data"].reset_index(level="time", inplace=True)
data["wind"]["proc_data"].rename(columns={"time": "ds", data["wind"]["target"]: "y"}, inplace=True)

train, test = wind_model.split_df(data["wind"]["proc_data"], valid_p=0.3, freq="h")
metrics = wind_model.fit(train, validation_df=test,  freq="h", plot_live_loss=True)

data["wind"]["model"] = wind_model

metrics.tail(1)
```


    
![png](TSDB%202022%20-%20ProjectWork_files/TSDB%202022%20-%20ProjectWork_15_0.png)
    


    Epoch[137/137]: 100%|█| 137/137 [01:00<00:00,  2.25it/s, SmoothL1Loss=0.00722, MAE=0.289, RMSE=0.39, RegLoss=0, MAE_val
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SmoothL1Loss</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>RegLoss</th>
      <th>SmoothL1Loss_val</th>
      <th>MAE_val</th>
      <th>RMSE_val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>136</th>
      <td>0.007224</td>
      <td>0.289203</td>
      <td>0.390305</td>
      <td>0.0</td>
      <td>0.048525</td>
      <td>0.879514</td>
      <td>1.002501</td>
    </tr>
  </tbody>
</table>
</div>




```python
for value in data.values():
    value["forecast"] = value["model"].predict(value["proc_data"])
    value["model"].plot(value["forecast"], figsize=(40*cm, 9*cm))
    plt.show()
```

    WARNING - (NP.plotting.plot) - Legend is available only for the ten first handles
    


    
![png](TSDB%202022%20-%20ProjectWork_files/TSDB%202022%20-%20ProjectWork_16_1.png)
    


    WARNING - (NP.plotting.plot) - Legend is available only for the ten first handles
    


    
![png](TSDB%202022%20-%20ProjectWork_files/TSDB%202022%20-%20ProjectWork_16_3.png)
    


It is evident how simple univariate forecasting performs poorly, since wind data, differently from solar, doesn't have a strong seasonality.  

### Forecasting components (*solar generation*)
In the following, we report the components of the forecasted curve. The trend curve has an explainable decreasing shape for January and February. Notice how the trend curve becomes almost flat after the beginning of March. This may be due to contingential weather conditions occurred in Spain during 2017. The daily trend, instead, seems to be correct, since it has a higher plateau during the middle of the day, when the sun provides the most irradiance.


```python
m.plot_components(data["solar"]["log_forecast"], figsize=(40*cm, 30*cm))
plt.show()
```


    
![png](TSDB%202022%20-%20ProjectWork_files/TSDB%202022%20-%20ProjectWork_19_0.png)
    


## Covariate Multivariate Forecasting 
<img src="img/darts.png" style="width: 250px; float: right;"/>


Darts offers a variety of models, from classics such as ARIMA to state-of-the-art deep neural networks. The emphasis of the library is on offering modern machine learning functionalities, such as supporting multidimensional series, meta-learning on multiple series, training on large datasets, incorporating external data, ensembling models, and providing rich support for probabilistic forecasting. The need for Darts comes from NeuralProphet lack of multivariate capabilities, which will prove essential to improve the quality of wind energy forecasting.

<!-- Darts offers a variety of models, from classics such as ARIMA to state-of-the-art deep neural networks. The emphasis of the library is on offering modern machine learning functionalities, such as supporting multidimensional series, meta-learning on multiple series, training on large datasets, incorporating external data, ensembling models, and providing a rich support for probabilistic forecasting [2]. The need for Darts come for the NeuralProphet lack of multivariate forecasting capabilities, which will prove essential to improve the wind forecasting quality. -->


```python
import torch

from darts import TimeSeries
from darts.utils.model_selection import train_test_split
from darts.models import BlockRNNModel
from darts.dataprocessing.transformers import Scaler

from darts.metrics import mae

# Random seeds
torch.manual_seed(22)
np.random.seed(22)
```

### Pulling and processing weather data 
Weather data, collected from OpenWeatherMap contains several fields, of which only `windSpeed` and `windDeg` will be used. Of course, one could try to integrate more attributes, but it will inevitably slow the training phase. Wind information will be used as past covariates for the wind generation curve.


```python
columns = [
    "time", "city",
    "windSpeed", "windDeg"
]

# Change back the datetime index from "ds" to "time"
data["wind"]["proc_data"].rename(columns={"ds": "time"}, inplace=True)
data["wind"]["proc_data"].set_index("time", inplace=True)
series_wind = TimeSeries.from_dataframe(data["wind"]["proc_data"], freq="h")

# Pull the weather collection from Mongo and set timedate datatype
weather_data = client.Energy.HourlyWeather.find()
weather_df = pd.DataFrame.from_records(weather_data, columns=columns)
weather_df["time"] = pd.to_datetime(weather_df['time'], utc=True).dt.tz_localize(None)
weather_df.set_index(keys=["time", "city"], inplace=True)

# Remove duplicate timedates
weather_df = weather_df[~weather_df.index.duplicated(keep='first')]
```


```python
weather_df = weather_df.query(f'"{interval_start}" <= time <= "{interval_end}"')
weather_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>windSpeed</th>
      <th>windDeg</th>
    </tr>
    <tr>
      <th>time</th>
      <th>city</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-01-01 00:00:00</th>
      <th>Valencia</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2017-01-01 01:00:00</th>
      <th>Valencia</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2017-01-01 02:00:00</th>
      <th>Valencia</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2017-01-01 03:00:00</th>
      <th>Valencia</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2017-01-01 04:00:00</th>
      <th>Valencia</th>
      <td>2</td>
      <td>260</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-06-30 20:00:00</th>
      <th>Seville</th>
      <td>2</td>
      <td>350</td>
    </tr>
    <tr>
      <th>2017-06-30 21:00:00</th>
      <th>Seville</th>
      <td>1</td>
      <td>330</td>
    </tr>
    <tr>
      <th>2017-06-30 22:00:00</th>
      <th>Seville</th>
      <td>4</td>
      <td>337</td>
    </tr>
    <tr>
      <th>2017-06-30 23:00:00</th>
      <th>Seville</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2017-07-01 00:00:00</th>
      <th>Seville</th>
      <td>5</td>
      <td>348</td>
    </tr>
  </tbody>
</table>
<p>21725 rows × 2 columns</p>
</div>




```python
# Extract the cities list
cities = weather_df.index.get_level_values("city").unique().tolist()
cities
```




    ['Valencia', 'Madrid', 'Bilbao', ' Barcelona', 'Seville']



Weather data is related to different Spanish cities. Since we have the aggregate amount of wind energy produced, we will use the `windSpeed` and `windDeg` of each city, creating one field for each attribute.


```python
weather_cov_df = pd.DataFrame({"time": pd.date_range(interval_start, interval_end, freq="h")}) \
                             .set_index("time")
for city in cities:
    weather_city_df = weather_df.query(f"city=='{city}'") \
                                .reset_index(level="city", drop=True) \
                                .rename(columns={"windSpeed": f"windSpeed{city}", 
                                                 "windDeg": f"windDeg{city}"})
    weather_cov_df = weather_cov_df.join(weather_city_df, how="left")
    
weather_cov_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>windSpeedValencia</th>
      <th>windDegValencia</th>
      <th>windSpeedMadrid</th>
      <th>windDegMadrid</th>
      <th>windSpeedBilbao</th>
      <th>windDegBilbao</th>
      <th>windSpeed Barcelona</th>
      <th>windDeg Barcelona</th>
      <th>windSpeedSeville</th>
      <th>windDegSeville</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-01-01 00:00:00</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>350</td>
      <td>1</td>
      <td>178</td>
      <td>5</td>
      <td>260</td>
      <td>2</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2017-01-01 01:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>340</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>260</td>
      <td>2</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2017-01-01 02:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>340</td>
      <td>1</td>
      <td>175</td>
      <td>5</td>
      <td>260</td>
      <td>2</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2017-01-01 03:00:00</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>175</td>
      <td>5</td>
      <td>260</td>
      <td>2</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2017-01-01 04:00:00</th>
      <td>2</td>
      <td>260</td>
      <td>1</td>
      <td>350</td>
      <td>1</td>
      <td>175</td>
      <td>5</td>
      <td>260</td>
      <td>0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-06-30 20:00:00</th>
      <td>4</td>
      <td>60</td>
      <td>3</td>
      <td>330</td>
      <td>3</td>
      <td>270</td>
      <td>4</td>
      <td>60</td>
      <td>2</td>
      <td>350</td>
    </tr>
    <tr>
      <th>2017-06-30 21:00:00</th>
      <td>4</td>
      <td>100</td>
      <td>6</td>
      <td>280</td>
      <td>4</td>
      <td>290</td>
      <td>4</td>
      <td>40</td>
      <td>1</td>
      <td>330</td>
    </tr>
    <tr>
      <th>2017-06-30 22:00:00</th>
      <td>2</td>
      <td>360</td>
      <td>3</td>
      <td>320</td>
      <td>3</td>
      <td>270</td>
      <td>1</td>
      <td>40</td>
      <td>4</td>
      <td>337</td>
    </tr>
    <tr>
      <th>2017-06-30 23:00:00</th>
      <td>2</td>
      <td>10</td>
      <td>5</td>
      <td>350</td>
      <td>2</td>
      <td>300</td>
      <td>2</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2017-07-01 00:00:00</th>
      <td>2</td>
      <td>20</td>
      <td>7</td>
      <td>360</td>
      <td>2</td>
      <td>270</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>348</td>
    </tr>
  </tbody>
</table>
<p>4345 rows × 10 columns</p>
</div>



We can now transform the `Dataframe` into a `TimeSeries`, which is the Darts main class.


```python
weather_cov_ts = TimeSeries.from_dataframe(weather_cov_df, freq="h")
```


```python
# Split target and covariates in train and test data
series_wind_train, series_wind_test = train_test_split(series_wind, test_size=0.3)
weather_cov_ts_train, weather_cov_ts_test = train_test_split(weather_cov_ts, test_size=0.3)
```

### Training the model and forecasting
The selected model is the `BlockRNNModel`, which is a Recursive Neural Network with LSTM nodes. It is one of the models which allow for both covariate and multivariate forecasting.


```python
model = BlockRNNModel(
    model="LSTM",
    input_chunk_length=24,
    output_chunk_length=12,
    n_epochs=300,
    random_state=22,
)

model.fit(series=series_wind_train, 
          past_covariates=weather_cov_ts_train, 
          verbose=True)

today = datetime.now().strftime("%d-%m-%YT%H-%M-%S")
model.save_model(f"model/{today}.pth.tar")
```

    [2022-03-11 19:14:57,605] INFO | darts.models.forecasting.torch_forecasting_model | Train dataset contains 3007 samples.
    [2022-03-11 19:14:57,605] INFO | darts.models.forecasting.torch_forecasting_model | Train dataset contains 3007 samples.
    [2022-03-11 19:14:57,609] INFO | darts.models.forecasting.torch_forecasting_model | Time series values are 64-bits; casting model to float64.
    [2022-03-11 19:14:57,609] INFO | darts.models.forecasting.torch_forecasting_model | Time series values are 64-bits; casting model to float64.
    0%|          | 0/300 [00:00<?, ?it/s]
    Training loss: 0.11726


Given the resulting model, we can use it to evaluate the historical forecasts that would have been obtained by this model on the `series`. The metric used is MAE, as it did for the univariate case.


```python
series_wind_back = model.historical_forecasts(
    series=series_wind, 
    past_covariates=weather_cov_ts,
    retrain=False,
    start=0.6,
    forecast_horizon=24,
    verbose=True)

fig = plt.figure(figsize=(40*cm, 9*cm))
series_wind.plot(label="actual")
series_wind_back.plot(label="forecast", alpha=0.1)

print(f"Validation MAE = {round(mae(series_wind, series_wind_back),4)}")
```

    MAE = 0.4804
    


    
![png](TSDB%202022%20-%20ProjectWork_files/TSDB%202022%20-%20ProjectWork_36_1.png)
    


It can be seen that the performance of the `BlockRNNModel`, with multivarate past covariates, outperforms NeuralProphet: the former has a MAE of **0.4804**, while the latter is **0.768**.

## Conclusions and Future Work

<img src="img/future_works.png" style="width: 700px; float: right;"/>

The objective of this project work was to create the preliminary basis for a forecasting tool for renewable energy generation.
This project gives a glimpse of the difficulties of predicting curves that do not have inherent seasonality patterns, such as wind generation. 

<img src="img/serra_giannina.png" align="left" style="width: 400px;"/>

Future work would focus on deploying a pipeline for pulling data, regarding energy and weather, in order to build a renewable energy forecasting tool. Important features that can also be considered are the aggregate or local load and energy price.
In particular, several open-source tools, other than MongoDB, have been developed for timeseries management, such as [InfluxDB](https://www.influxdata.com/) and its plugin [Telegraph](https://www.influxdata.com/time-series-platform/telegraf/).
The following is a custom [TOML](https://github.com/toml-lang/toml) configuration file used to pull data from OpenWeather. In particular, the current latitude and longitude coordinates target the "Serra Giannina" eolic plant, which is located in Apulia, near Melfi. This is being used in place of the default Telegraph OpenWeather `.conf` file, since the latter is too simplistic. 

```
# [global_tags]
user = "${USER}"

[[outputs.influxdb_v2]]
urls = ["http://localhost:8086"]
token = "${INFLUX_TOKEN}"
organization = "influx-org"
bucket = "openweather"

# OpenWeatherMap API (Custom)
[[inputs.http]]
urls=["https://api.openweathermap.org/data/2.5/weather?lat=41.01&lon=15.69&appid=${OPENWEATHER_API_KEY}&units=metric"]
method = "GET" 
data_format = "json_v2"
interval = "5s"
tagexclude = ["url", "host", "user"]

	[[inputs.http.json_v2]]
	measurement_name = "openweather"
	timestamp_path = "dt"
	timestamp_format = "unix"
	timestamp_timezone = "UTC"
	
		[[inputs.http.json_v2.field]]
		path = "coord.lon"
		type = "float"
		
		[[inputs.http.json_v2.field]]
		path = "coord.lat"
		type = "float"
		
		[[inputs.http.json_v2.tag]]
		path = "weather.#.main"
		rename = "weather_main"
		
		[[inputs.http.json_v2.tag]]
		path = "base"
		
		[[inputs.http.json_v2.object]]
		path = "main"
		type = "float"
		
		[[inputs.http.json_v2.field]]
		path = "visibility"
		type = "float"
		
		[[inputs.http.json_v2.object]]
		path = "wind"
		type = "float"
		
		[[inputs.http.json_v2.field]]
		path = "clouds.all"
		rename = "clouds"
		type = "float"
		
		[[inputs.http.json_v2.tag]]
		path = "sys.country"
		
		[[inputs.http.json_v2.tag]]
		path = "name"
```
## References 
Triebe, Oskar, et al. "NeuralProphet: Explainable Forecasting at Scale." arXiv preprint arXiv:2111.15397 (2021). 

Herzen, Julien, et al. "Darts: User-friendly modern machine learning for time series." arXiv preprint arXiv:2110.03224 (2021).

Banker, Kyle, et al. MongoDB in Action: Covers MongoDB version 3.0. Simon and Schuster, 2016.

Lütkepohl, Helmut, and Fang Xu. "The role of the log transformation in forecasting economic variables." Empirical Economics 42.3 (2012): 619-638.
