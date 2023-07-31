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
# For the dashboard, we need multiple CSV :
# data after featuring = the data from all clients we need to display on the dashboard
# main info client = just some informations about the clients before scaling / one hot encoding
# feature info = a short description of each features
# mean std on columns = to go back from scaled value to original values using mean/std

# We also need the explainer that is a bunch of graphics from dashexplainer from SHAP.
# We will use their graphics with some changes that will be done in the following functions.

df = pd.read_csv('data_after_featuring.csv.gz').drop(columns = ['Unnamed: 0'])
df2 = pd.read_csv('main_info_client.csv').drop(columns = ['Unnamed: 0'])
dff = pd.read_csv('features_info.csv').drop(columns = ['Unnamed: 0'])
feature_scale = pd.read_csv('mean_std_on_columns.csv').drop(columns = ['Unnamed: 0'])
data_clean = df2.copy()
data_clean = data_clean[[a for a in df2.columns if ((df2[a].dtype == 'float')|(df2[a].dtype == 'int64'))]]
data_clean = data_clean.dropna(axis=0, how='any')
explainer = ClassifierExplainer.from_file('explainer.joblib')


# We will need similar clients in the dashboard. To find them, we will use a nearest neighbors with the clients main infos :

scaler = preprocessing.StandardScaler()
scaler.fit(data_clean[[a for a in df2.columns if (df2[a].dtype == 'float')]])
data_clean[[a for a in df2.columns if (df2[a].dtype == 'float')]] = scaler.transform(data_clean[[a for a in df2.columns if (df2[a].dtype == 'float')]])
nei = NearestNeighbors(n_neighbors = 10)
nei.fit(data_clean[[a for a in df2.columns if (df2[a].dtype == 'float')]])

def find_similar_client(client_id):
    """
    The purpose of this function is to dispay a dataframe of similar clients from an unique client.
    Parameter
    ----------
    client_id (int or string number) : the client id from "SK_CURR_ID" in data_after_featuring.csv
    
    Returns
    ----------
    return_df (pandas dataframe) : a dataframe with 10 clients sharing similar features
    """
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




def get_precision(client_id):
    
    """
    To create dropdowns, we want the option to be in the same order than feature importance by shap values.
    Since we use local shap values for 1 specific client, we create a function that get the dict of thoose values.
    We use here the file explainer.joblib as "explainer" as global variable.
    
    Parameters
    ----------
    client_id (int or str) : client identification number from 'SK_CURR_ID' feature.
    """
    
    index_client = int(df.loc[df['SK_ID_CURR'] == int(client_id)].index[0])
    df_importance = explainer.get_contrib_df(index = index_client)
    df_importance['contribution'] = abs(df_importance['contribution'])
    df_importance = df_importance.sort_values(by = 'contribution', ascending = False)
    df_importance = df_importance.loc[(df_importance['col'] != '_PREDICTION')]
    df_importance = df_importance.loc[(df_importance['col'] != '_BASE')]
    ordered_list = list(df_importance['col'])
    return [{'label':i, 'value':i} for i in ordered_list]



def client_position(id_client):
    
    """
    Provide an histogramm of all clients predicted probability. Gives the current client position on the hist.
    
    Parameters
    ----------
    id_client (int or str) : client identification number from 'SK_CURR_ID' feature.
    """
    
    
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
    
    """
    We will use explainer.joblib created on Projet7-gridsearchs-on-chosen-model (as 'explainer' in this notebook) BUT
    we used already scaled values to create it (so it .predict / .predict_proba correctly).
    This function takes the 'explainer.plot_contribution' graph from explainerdashboard and rescale the values from normalised
    to starting values.
    
    Parameters
    ----------
    -ind (int) : index of the client (real index, not 'SK_CURR_ID')
    -tpox (int) : number of feature to take in consideration in explainer.plot_contribution
    -feature_data_scaler (panda DataFrame) : use 'mean_std_on_columns.csv' as 'feature_data_scaler'
    
    Return
    ----------
    -Graph (ploty graph) : explainer.plot_contributions with updated values of the features.
    """
    
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
    
    """
    We will use explainer.joblib created on Projet7-gridsearchs-on-chosen-model (as 'explainer' in this notebook) BUT
    we used already scaled values to create it (so it .predict / .predict_proba correctly).
    This function takes the 'explainer.plot_pdp' graph from explainerdashboard and rescale the values from normalised
    to starting values.
    
    Parameters
    ----------
    -feature (string) : feature we want to be displayed in the plit.pdp
    -ind (int) : index of the client (real index, not 'SK_CURR_ID')
    -feature_data_scaler (panda DataFrame) : use 'mean_std_on_columns.csv' as 'feature_data_scaler'
    
    Return
    ----------
    -Graph (ploty graph) : explainer.plot_pdp with updated values of the features.
    """
    
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
    """
    Create an histogram of the distribution of a features along all datas. 
    df is global variable 'data_after_featuring.csv'.
    feature_scale is global variable 'mean_std_on_columns.csv'
    
    Parameter
    ----------
    -id_client (int or string) : client identification number from 'SK_CURR_ID' feature.
    -feature (string) : choosen feature in 'data_after_featuring.csv' columns
    -ignore_high_value(Bool, default = False) : to ignore outliers (highest values) or not.
    """
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


# App callbacks

# First callback : When clicking on submit button -> 
#                        -show client predict.proba position in an histogramm of all clients
#                        -show client general informations 
#                        -Show client ability to pay the loan (able/unable)
#                        -maxe 2 buttons visible : show similar clients / Show decision explanation

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

# When the user click on the button "show similar clients" it display :
#                        -A table with 10 similar clients and their prediction able/unable
#                        -A dropdown to choose a feature

    
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

# After choosing a feature in the dropdown, it will show :
#                  -A short description of the feature
#                  -An histogramm of the distribution of the feature on all clients, with current client position.
    
    
    
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

# This callback is applied when clicking on 'get info on predictions' and display :
#                  -The explainer.plot_contribution with shap values for the client
#                  -A dropdown to choose a feature to display pdp plot.
    
    
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
    
    
# This last callback apply the dropdown value choosen on previous callback and shows :
#                  -The information about the choosen feature
#                  -the explained.plot_pdp of the selected feature.
    
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

