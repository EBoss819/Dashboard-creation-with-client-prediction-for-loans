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
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import requests
from sklearn.neighbors import NearestNeighbors
from explainerdashboard import ClassifierExplainer
import plotly.graph_objects as go
import plotly.io as pio
import math
import plotly.express as px


sns.set_theme(style='darkgrid', palette='dark')

# +
df = pd.read_csv('data_after_featuring.csv.gz').drop(columns = ['Unnamed: 0'])
df2 = pd.read_csv('main_info_client.csv').drop(columns = ['Unnamed: 0'])
dff = pd.read_csv('features_info.csv').drop(columns = ['Unnamed: 0'])
feature_scale = pd.read_csv('mean_std_on_columns.csv').drop(columns = ['Unnamed: 0'])
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



def client_position(id_client):
    data = explainer.get_precision_df()
    index_client = int(df.loc[df['SK_ID_CURR'] == int(id_client)].index[0])
    proba_client = explainer.pred_probas()[index_client]
    data['p_min'] = data['p_min'].apply(lambda x : x + 0.05)
    fig = dict({
        "data": [{"type": "bar",
                  "x": data['p_min'],
                  "y": data['count']}],
        "layout": {"title": {"text": "Distribution of clients by predicted probability"}}
    })
    fig = go.Figure(fig)
    fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 0,
        dtick = 0.1)
    )
    reference_line = go.Scatter(x=[proba_client, proba_client],
                                y=[-1000, 1000],
                                mode="lines",
                                line=go.scatter.Line(color="black"),
                                showlegend=False)
    fig.add_annotation(dict(font=dict(color='black',size=15),
                                            x= proba_client,
                                            y=1000,
                                            text=f"client {id_client} with {round(proba_client,3)}"),
                      bgcolor = 'white')
    fig.add_trace(reference_line)
    return fig



def update_contrib(ind, tpox, feature_data_scaler):
    graph = explainer.plot_contributions(index = ind, topx = tpox).to_dict() 
    new_text = []
    new_text.append(graph['data'][1]['text'][0])
    for string in graph['data'][1]['text'][1:-2]:
        feature = string.split('=', 1)[0]
        if feature in feature_scale.columns:
            mean = feature_data_scaler.loc[0,feature]
            std = feature_data_scaler.loc[1,feature] 
            value = float(string.split('=', 1)[1].split('<BR>', 1)[0])*std+mean
        else : 
            value = float(string.split('=', 1)[1].split('<BR>', 1)[0])
        shapvalue  = float(string.split('=', 1)[1].split('<BR>', 1)[1])
        new_text.append(f'{feature} = {value} <BR> impact : {shapvalue}')
    new_text.append(graph['data'][1]['text'][-2])
    new_text.append(graph['data'][1]['text'][-1])
    graph['data'][1]['text'] = new_text
    return graph



def rescale_pdp(feature, ind, feature_data_scaler):
    graph = explainer.plot_pdp(feature, index = ind).to_dict()
    for i in graph['data']:
        if feature in feature_data_scaler.columns:
            mean = feature_data_scaler.loc[0,feature]
            std = feature_data_scaler.loc[1,feature]            
            i['x'] = i['x']*std+mean
    if feature in feature_data_scaler.columns:
        graph['layout']['annotations'][0]['x'] = graph['layout']['annotations'][0]['x']*std+mean
        graph['layout']['shapes'][0]['x0'] = graph['layout']['shapes'][0]['x0']*std+mean
        graph['layout']['shapes'][0]['x1'] = graph['layout']['shapes'][0]['x1']*std+mean
        graph['layout']['shapes'][1]['x0'] = graph['layout']['shapes'][1]['x0']*std+mean
        graph['layout']['shapes'][1]['x1'] = graph['layout']['shapes'][1]['x1']*std+mean
    client_value = graph['layout']['annotations'][0]['x']
    graph['layout']['annotations'][0]['text'] = f'client_value : {round(client_value,3)}'
    graph['layout']['annotations'][1]['x'] = 0
    return graph




def position_client_feature(id_client, feature, ignore_high_value = False):
    copy = df.copy()
    copy = copy.sort_values(feature, ascending = False)
    if feature in feature_scale.columns:
        mean = feature_scale.loc[0,feature]
        std = feature_scale.loc[1,feature]
        copy[feature] = copy[feature]*std+mean
    else :
        mean = 0
        std = 1
    if ignore_high_value:
        fig = px.histogram(copy.iloc[10:], feature)
    else:
        fig = px.histogram(copy, feature)
    value_client = int(df.loc[df['SK_ID_CURR'] == int(id_client)][feature]*std+mean)
    fig.add_annotation(dict(font=dict(color='black',size=15),
                                            x= value_client,
                                            y= 0,
                                            text=f"client {id_client} with {round(value_client, 2)}"),
                      bgcolor = 'white')
    del copy
    return fig


# +
# Afficher la prédiction d'un client via son ID
# Afficher la description du client via son ID
# Comparer le client avec des clients similaires
from dash_bootstrap_templates import load_figure_template

# Initialize the app
app = Dash(external_stylesheets=[dbc.themes.MORPH])
load_figure_template('MORPH')

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
             ],
        style={'textAlign': 'center'}), 
    
    html.Br(),
    
    html.H3(id='output_div_1'),
    html.Div([
        dash_table.DataTable(id='tbl', style_table={'overflowX': 'auto'})
             ],
        style={'margin-right' : '100px', 'margin-left' : '100px'}),
    
    html.Br(),
    
    dcc.Graph(figure = {},
          id = 'distribution',
          style = {'display' : 'none'}),
    
    html.Br(),
    
    html.H2(id='output_div_2', style = {'textAlign': 'center'}),
    
    html.Div([
        html.Button(id='info_on_feature',        
                children='get infos on prediction',
                style={'textAlign': 'center'})
         ],
        id = 'info_on_feature_button',
        style={'display': 'none', 'float': 'right','margin': 'auto'}),
    
    html.Div([
        html.Button(id='similar_client_button',        
                children='compare with other clients',
                style={'textAlign': 'center'})
            ],
        id = 'similar_client_button_hide',
        style={'display': 'none', 'float': 'right','margin': 'auto'}),
    
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    
    html.H3(['Here is some similar clients :'],
     id = 'similar_clients',
     style = {'display' : 'none', 'textAlign' : 'center'}),
    
    html.H4(['Thoose clients share multiple similar caracteristics with current client.'],
            id = 'similar_clients_text',
            style = {'display' : 'none'}),
    
    html.Div([dash_table.DataTable(id='tbl2',style_table={'overflowX': 'auto'})],
             id = 'hide_and_show_similar_table',
             style={'display': 'none'}),
    
    html.Br(),
    html.H4(['Choose a feature to see position of current client compared to all database'],
            id = 'similar_clients_text_dropdown',
            style = {'display' : 'none'}),
    html.Br(),
    html.Div([dcc.Dropdown(
        id='dropdown_clients',
        options=[{}])
             ],
             id = 'hide_and_show_dropdown_client',
             style = {'display' : 'none'}
            ),
    
    html.H3(id = 'other_info_feature',
         style = {'display' : 'none'}),
    
    
    dcc.Graph(figure = {},
      id = 'feature_distrib',
      style = {'display' : 'none'}),
    
    html.Br(),
    
    html.H3(['Importance of each feature from highest impact to lowest'],
             id = 'slider_text',
             style = {'display' : 'none'}),
    
    html.Div([dcc.Slider(
            id = 'slider',
            min=0,
            max=435,
            value=10,
            
        )],
             id = 'hide_and_show_slider',
             style = {'display' : 'none'}),
    
    dcc.Graph(figure = {},
              id = 'graph',
              style = {'display' : 'none'}),
    html.Br(),
    
    html.H3(['Choose a feature to see the Partial Dependance Plot (PDD)'],
     id = 'dropout_text',
     style = {'display' : 'none'}),
    html.H5(['PDD are used to show variation on the prediction when all other feature are fixed, only checking what would happend with a different value on the selected feature'],
     id = 'pdd_explanation',
     style = {'display' : 'none'}),
    
    html.Div([dcc.Dropdown(
        id='dropdown',
        options=[{}])
             ],
             id = 'hide_and_show_dropdown',
             style = {'display' : 'none'}
            ),
    
    html.Br(),
    
    html.H4("Information about the feature :",
            id='feature_info_hide',
            style = {'display' : 'none'}),
    
    html.H5(id='feature_info',
            style = {'display' : 'none'}),
    
    html.Br(),
    
    dcc.Graph(figure = {},
          id = 'graph2',
              style = {'display' : 'none'}),
    
    html.Br(),
    
    html.H4("Select a second feature :",
            id='double_feature_info_hide',
            style = {'display' : 'none'}),  
    
    html.Br(),
    
    html.Div([dcc.Dropdown(
    id='dropdown2',
    options=[{}])
         ],
         id = 'hide_and_show_dropdown2',
         style = {'display' : 'none'}
        ),
    
    html.Br(),
    dcc.Graph(figure = {},
              id = 'graph3',
              style = {'display' : 'none'}),

    html.Br(),
    html.Br(),

])

@app.callback(Output('output_div_1','children'),
              Output('output_div_1','style'),
              Output('tbl', 'data'),
              Output('output_div_2', 'children'),
              Output('similar_client_button_hide', 'style'),
              Output('info_on_feature_button', 'style'),
              Output('distribution', 'figure'),
              Output('distribution', 'style'),
              Input('textbox', 'value'),
              Input('slider', 'value'),
              Input('submit-button','n_clicks'),
              State('submit-button','n_clicks'))
def update_datatable(id_client, slide, n_clicks,state):            
    if n_clicks:                            
        data = df2.loc[df2['SK_ID_CURR'] == int(id_client)]
        data = data.to_dict('records') 
        tested_client = df.loc[df['SK_ID_CURR'] ==  int(id_client)].drop(columns = ['SK_ID_CURR', 'TARGET'])
        feature_client = tested_client.to_dict('records')[0]
        prediction = requests.post('http://127.0.0.1:8000/predict', json = feature_client).json()
        
        return (f'Infos on client n°{id_client} :',
                {'textAlign' : 'center'},
                data,
                f'Predicted capacity to pay back the loan : {prediction["prediction"]}.',
                {'display' : 'block', 'float': 'left','margin': '100px'},
                {'display': 'block', 'float': 'right','margin': '100px'},
                client_position(id_client),
                {'display' : 'block', 'margin-right' : '100px', 'margin-left' : '100px'})
   

    
@app.callback(Output('tbl2', 'data'),
              Output('similar_clients', 'style'),
              Output('hide_and_show_similar_table', 'style'),
              Output('similar_client_button', 'children'),
              Output('hide_and_show_dropdown_client', 'style'),
              Output('dropdown_clients', 'options'),
              Output('similar_clients_text', 'style'),
              Output('similar_clients_text_dropdown', 'style'),
              Input('textbox', 'value'),
              Input('similar_client_button_hide','n_clicks'),
              State('similar_client_button_hide','n_clicks'))
def client_info(id_client, n_clicks, state):            
    if n_clicks%2 == 1:                            
        similar = find_similar_client(id_client)
        similar = similar.to_dict('records')
        return (similar,
                {'display' : 'block', 'textAlign' : 'center'},
                {'display' : 'block', 'margin-right' : '100px', 'margin-left' : '100px'},
                'compare with other clients (click to hide/show)',
                {'display' : 'block', 'margin-right' : '100px', 'margin-left' : '100px'},
                get_precision(id_client),
                {'display' : 'block', 'textAlign' : 'center'},
                {'display' : 'block', 'margin-left' : '100px'}
               )
    if n_clicks%2 == 0:    
        similar = find_similar_client(id_client)
        similar = similar.to_dict('records')
        return (similar,
                {'display' : 'none'},
                {'display' : 'none'},
                'compare with other clients (click to hide/show)',
                {'display' : 'none'},
                get_precision(id_client),
                {'display' : 'none'},
                {'display' : 'none'}
               )


    
    
    
@app.callback(Output('feature_distrib','style'),
              Output('feature_distrib','figure'),
              Output('other_info_feature', 'children'),
              Output('other_info_feature', 'style'),
              Input('dropdown_clients', 'value'),
              Input('textbox', 'value'),
              Input('similar_client_button_hide','n_clicks'),
              State('similar_client_button_hide','n_clicks')
             )
def featurestuff(dropdown, id_client, n_clicks, state):
    if n_clicks%2 == 1:   
        return ({'display' : 'block', 'margin-right' : '100px', 'margin-left' : '100px'},
                position_client_feature(id_client, dropdown, True),
                dff.loc[dff['feature'] == dropdown].iloc[0,1],
                {'display' : 'block', 'margin-right' : '100px', 'margin-left' : '150px'}
               )
    if n_clicks%2 == 0:   
        return ({'display' : 'none'},
                position_client_feature(id_client, dropdown, True),
                dff.loc[dff['feature'] == dropdown].iloc[0,1],
                {'display' : 'none'}
               )

    
    
@app.callback(Output('graph', 'figure'),
              Output('graph', 'style'),
              Output('slider_text', 'style'),
              Output('hide_and_show_slider', 'style'),
              Output('hide_and_show_dropdown', 'style'),
              Output('dropout_text', 'style'),
              Output('dropdown', 'options'),
              Output('info_on_feature', 'children'),
              Output('pdd_explanation', 'style'),
              Input('textbox', 'value'),
              Input('slider', 'value'),
              Input('info_on_feature_button','n_clicks'),
              State('info_on_feature_button','n_clicks'))
def feature_info(id_client, slide, n_clicks, state):            
    if n_clicks%2 == 1:    
        index_client = int(df.loc[df['SK_ID_CURR'] == int(id_client)].index[0])
        return (update_contrib(index_client, slide, feature_scale),
                {'display' : 'block', 'margin-right' : '100px', 'margin-left' : '100px'},
                {'display' : 'block', 'textAlign' : 'center'},
                {'display' : 'block', 'margin-right' : '150px', 'margin-left' : '150px'},
                {'display' : 'block', 'margin-right' : '150px', 'margin-left' : '150px'},
                {'display' : 'block', 'textAlign' : 'center'},
                get_precision(id_client),
               'get infos on prediction (click to hide/show)',
                {'display' : 'block', 'textAlign' : 'center'})
    if n_clicks%2 == 0: 
        index_client = int(df.loc[df['SK_ID_CURR'] == int(id_client)].index[0])
        return (update_contrib(index_client, slide, feature_scale),
                {'display' : 'none'},
                {'display' : 'none'},
                {'display' : 'none'},
                {'display' : 'none'},
                {'display' : 'none'},
                get_precision(id_client),
                'get infos on prediction (click to hide/show)',
                {'display' : 'none'})
    
    
    
@app.callback(Output('graph2','style'),
              Output('graph2','figure'),
              Output('feature_info_hide', 'style'),
              Output('feature_info','children'),
              Output('feature_info','style'),
              Input('dropdown', 'value'),
              Input('textbox', 'value'),
              Input('info_on_feature_button','n_clicks'),
              State('info_on_feature_button','n_clicks'))
def do_stuff(dropdown, id_client, n_clicks, state):
    if n_clicks%2 == 1:   
        index_client = int(df.loc[df['SK_ID_CURR'] == int(id_client)].index[0])
        feature_info = dff.loc[dff['feature'] == dropdown].iloc[0,1]
        return ({'display' : 'block', 'margin-right' : '100px', 'margin-left' : '100px'},
                rescale_pdp(dropdown, index_client, feature_scale),
                {'display' : 'block', 'margin-left' : '100px'},
                feature_info,
                {'display' : 'block', 'margin-left' : '100px'}
               )
    if n_clicks%2 == 0:   
        index_client = int(df.loc[df['SK_ID_CURR'] == int(id_client)].index[0])
        feature_info = dff.loc[dff['feature'] == dropdown].iloc[0,1]
        return ({'display' : 'none'},
                rescale_pdp(dropdown, index_client, feature_scale),
                {'display' : 'none'},
                feature_info,
                {'display' : 'none'}
               )


    

# Run the app
if __name__ == '__main__':
    app.run_server(port = 8050, host = '0.0.0.0', debug=False) 
# -

