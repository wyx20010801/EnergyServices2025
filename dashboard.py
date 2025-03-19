import pickle
import dash
from dash import dcc,html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from sklearn import metrics
import numpy as np

df = pd.read_excel('D:/电力预测/forecast_data.xlsx')
df['Date'] = pd.to_datetime(df['Date'])  # create a new column 'data time' of datetime type
df2 = df.iloc[:, 1:8]
X2 = df2.values
# 使用 melt 将多列数据转换为长格式
df_melted = df.melt(id_vars=['Date'], value_vars=df.columns[1:8], var_name='variable', value_name='value')
# 绘制叠加折线图
fig1 = px.line(df_melted, x='Date', y='value', color='variable', title='Multi-line overlay chart')
# fig1.show()
df_real = pd.read_excel('D:/电力预测/testData_2019_SouthTower.xlsx')
y2 = df_real['South Tower (kWh)'].values

# load RF model
with open('D:/电力预测/south_tower_model_hour.plk', "rb") as f:
    RF_model2 = pickle.load(f)

y2_pred_RF = RF_model2.predict(X2)
#Evaluate errors
MAE_RF = metrics.mean_absolute_error(y2, y2_pred_RF)
MBE_RF = np.mean(y2 - y2_pred_RF)
MSE_RF = metrics.mean_squared_error(y2, y2_pred_RF)
RMSE_RF = np.sqrt(metrics.mean_squared_error(y2, y2_pred_RF))
cvRMSE_RF = RMSE_RF/np.mean(y2)
NMBE_RF = MBE_RF/np.mean(y2)
# print("MAE_RF:",MAE_RF)
# print("MBE_RF:",MBE_RF)
# print("MSE_RF:",MSE_RF)
# print("RMSE_RF:",RMSE_RF)
# print("cvRMSE_RF:",cvRMSE_RF)
# print("NMBE_RF:",MBE_RF)


# load LSTM model
with open('D:/电力预测/LR_model.plk', "rb") as f:
    LR_model2 = pickle.load(f)
y2_pred_LR = LR_model2.predict(X2)
#Evaluate errors
MAE_LR = metrics.mean_absolute_error(y2, y2_pred_LR)
MBE_LR = np.mean(y2 - y2_pred_LR)
MSE_LR = metrics.mean_squared_error(y2, y2_pred_LR)
RMSE_LR = np.sqrt(metrics.mean_squared_error(y2, y2_pred_LR))
cvRMSE_LR = RMSE_LR/np.mean(y2)
NMBE_LR = MBE_LR/np.mean(y2)
# print("MAE_LR:",MAE_LR)
# print("MBE_LR:",MBE_LR)
# print("MSE_LR:",MSE_LR)
# print("RMSE_LR:",RMSE_LR)
# print("cvRMSE_LR:",cvRMSE_LR)
# print("NMBE_LR:",MBE_LR)

# Create data frames with prediction results and error metrics
d={'Methods':['Random Forest','LR'],'MAE':[MAE_RF,MAE_LR],'MBE':[MBE_RF,MBE_LR],'MSE':[MSE_RF,MSE_LR],'RMSE':[RMSE_RF,RMSE_LR],'cvRMSE':[cvRMSE_RF,cvRMSE_LR],'NMBE':[NMBE_RF,NMBE_LR]}
df_metrics = pd.DataFrame(data=d)
d={'Date':df_real['Date'].values, 'RandomForest':y2_pred_RF,'LR':y2_pred_LR}
df_forecast = pd.DataFrame(data=d)

# merge real and forecast results and creantes a figure with it
df_results = pd.merge(df_real,df_forecast,on='Date')
fig2 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:12])
# fig2.show()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H2('IST Energy Forecast tool (kWh)'),
    dcc.Tabs(id='tabs',value='tab-1',children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])

# Define auxiliary functions
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

@app.callback(Output('tabs-content', 'children'),Input('tabs','value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('IST Raw Data'),
            dcc.Graph(
                id='yearly-data',
                figure=fig1,
            ),
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('IST Electricity Forecast (kWh)'),
            dcc.Graph(
                id='yearly-data',
                figure=fig2,
            ),
            generate_table(df_metrics)
        ])
import threading
import time

def open_browser():
    time.sleep(2)  # 等待应用启动
    import webbrowser
    webbrowser.open('http://127.0.0.1:8050')

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()  # 异步打开浏览器
    app.run_server(debug=True, use_reloader=False)  # 启动Dash应用

