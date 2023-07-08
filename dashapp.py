# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer, auc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import requests
from sklearn.neighbors import NearestNeighbors
from explainerdashboard import ClassifierExplainer

sns.set_theme(style='darkgrid', palette='dark')

# +
df = pd.read_csv('data_after_featuring.csv.gz').drop(columns = ['Unnamed: 0'])
df2 = pd.read_csv('main_info_client.csv').drop(columns = ['Unnamed: 0'])
dff = pd.read_csv('features_info.csv').drop(columns = ['Unnamed: 0'])
data_clean = df2.copy()
data_clean = data_clean[[a for a in df2.columns if ((df2[a].dtype == 'float')|(df2[a].dtype == 'int64'))]]
data_clean = data_clean.dropna(axis=0, how='any')
scaler = preprocessing.StandardScaler()
scaler.fit(data_clean[[a for a in df2.columns if (df2[a].dtype == 'float')]])
data_clean[[a for a in df2.columns if (df2[a].dtype == 'float')]] = scaler.transform(data_clean[[a for a in df2.columns if (df2[a].dtype == 'float')]])
nei = NearestNeighbors(n_neighbors = 10)
nei.fit(data_clean[[a for a in df2.columns if (df2[a].dtype == 'float')]])

def find_similar_client(client_id):
    client = data_clean.loc[data_clean['SK_ID_CURR'] == int(client_id)]
    similar_clients = nei.kneighbors(client.drop(columns = ['SK_ID_CURR', 'CNT_CHILDREN']))
    client_id_similar = list(data_clean.iloc[list(similar_clients[1][0]),:]['SK_ID_CURR'])[1:]
    list_clients = df.loc[df['SK_ID_CURR'].isin(client_id_similar)].iloc[:,2:]
    return_df = df2.loc[df2['SK_ID_CURR'].isin(client_id_similar)]
    l = []
    for i in range(len(list_clients)):
        d = pd.DataFrame(list_clients.iloc[i,:]).T.to_dict('records')[0]
        prediction = requests.post('http://127.0.0.1:8000/predict', json = d).json()
        l.append(prediction['prediction'])
    return_df.insert(1, "PREDICTION", l)
    return_df
    return return_df



explainer = ClassifierExplainer.from_file('explainer.joblib')

def get_precision(client_id):
    index_client = int(df.loc[df['SK_ID_CURR'] == int(client_id)].index[0])
    df_importance = explainer.get_contrib_df(index = index_client)
    df_importance['contribution'] = abs(df_importance['contribution'])
    df_importance = df_importance.sort_values(by = 'contribution', ascending = False)
    df_importance = df_importance.loc[(df_importance['col'] != '_PREDICTION')]
    df_importance = df_importance.loc[(df_importance['col'] != '_BASE')]
    ordered_list = list(df_importance['col'])
    return [{'label':i, 'value':i} for i in ordered_list]

# +
# Afficher la prédiction d'un client via son ID
# Afficher la description du client via son ID
# Comparer le client avec des clients similaires


# Initialize the app
app = Dash(__name__)

# App layout

app.layout = html.Div([
    html.H1("Dashboard for client prediction and explanations", style={'textAlign': 'center'}),
    html.H3("Enter client id and submit", style={'textAlign': 'center'}),
    html.Div([
        dcc.Input(
            type = 'text',
            value = 100013,
            persistence=True,
            id = 'textbox',
            style={'textAlign': 'center'}
        )], style={'textAlign': 'center'}),
    html.Div([
        html.Button(id='submit-button',        
                    children='Submit',
                    style={'textAlign': 'center'})
             ], style={'textAlign': 'center'}), 
    html.Br(),
    html.H3(id='output_div_1'),
    dash_table.DataTable(id='tbl'),
    html.Br(),
    html.H2(id='output_div_2', style = {'textAlign': 'center'}),
    html.H3(['Here is some similar clients :'],
         id = 'similar_clients',
         style = {'display' : 'none'}),
    html.Br(),
    dash_table.DataTable(id='tbl2'),
    html.Br(),
    html.H3(['Slider for number of feature to show in graph'],
             id = 'slider_text',
             style = {'display' : 'none'}),
    html.Div([dcc.Slider(
            id = 'slider',
            min=0,
            max=435,
            value=10,
            
        )],id = 'hide_and_show_slider', style = {'display' : 'none'}),
    dcc.Graph(figure = {},
              id = 'graph', style = {'display' : 'none'}),
    html.H3(['Choose a feature to see the Partial Dependance Plot (PDD)'],
     id = 'dropout_text',
     style = {'display' : 'none'}),
    html.Br(),
    html.Div([dcc.Dropdown(
        id='dropdown',
        options=[{}])
             ],
             id = 'hide_and_show_dropdown',
             style = {'display' : 'none'}
            ),
    html.Br(),
    html.H3("Information about the feature :",id='feature_info_hide',style = {'display' : 'none'}),
    html.H3(id='feature_info'),
    html.Br(),
    dcc.Graph(figure = {},
          id = 'graph2', style = {'display' : 'none'}),
    html.Br(),

    html.Br(),
    html.Br(),

])

@app.callback(Output('output_div_1','children'),
              Output('tbl', 'data'),
              Output('output_div_2', 'children'),
              Output('tbl2', 'data'),
              Output('graph', 'figure'),
              Output('graph', 'style'),
              Output('slider_text', 'style'),
              Output('hide_and_show_slider', 'style'),
              Output('similar_clients', 'style'),
              Output('hide_and_show_dropdown', 'style'),
              Output('dropout_text', 'style'),
              Output('dropdown', 'options'),
              Input('textbox', 'value'),
              Input('slider', 'value'),
              Input('submit-button','n_clicks'),
              State('submit-button','n_clicks'))

def update_datatable(id_client, slide, n_clicks,csv_file):            
    if n_clicks:                            
        data = df2.loc[df2['SK_ID_CURR'] == int(id_client)]
        data = data.to_dict('records') 
        tested_client = df.loc[df['SK_ID_CURR'] ==  int(id_client)].drop(columns = ['SK_ID_CURR', 'TARGET'])
        feature_client = tested_client.to_dict('records')[0]
        prediction = requests.post('http://127.0.0.1:8000/predict', json = feature_client).json()
        similar = find_similar_client(id_client)
        similar = similar.to_dict('records')
        index_client = int(df.loc[df['SK_ID_CURR'] == int(id_client)].index[0])
        
        return (f'Infos on client n°{id_client} :',
                data,
                f'Predicted capacity to pay back the loan : {prediction["prediction"]}.',
                similar,
                explainer.plot_contributions(index = index_client, topx = int(slide)),
                {'display' : 'block'},
                {'display' : 'block'},
                {'display' : 'block'},
                {'display' : 'block'},
                {'display' : 'block'},
                {'display' : 'block'},
                get_precision(id_client))

@app.callback(Output('graph2','style'),
              Output('graph2','figure'),
              Output('feature_info_hide', 'style'),
              Output('feature_info','children'),
              Input('dropdown', 'value'),
              Input('textbox', 'value'))
def do_stuff(dropdown, id_client):
    index_client = int(df.loc[df['SK_ID_CURR'] == int(id_client)].index[0])
    feature_info = dff.loc[dff['feature'] == dropdown].iloc[0,1]
    return ({'display' : 'block'},
            explainer.plot_pdp(dropdown, index = index_client),
            {'display' : 'block'},
            feature_info
           )

    

# Run the app
if __name__ == '__main__':
    app.run_server(host = '0.0.0.0', port = 8050, debug=False)
