import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State, ALL
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

engine = create_engine('postgresql://postgres:D4t4B4s3@localhost:5432/fitness_database')
df = pd.read_sql('SELECT * FROM exercises', engine)
df['date'] = pd.to_datetime(df['date'])

sidebar = html.Div([
            html.H2('Sidebar'),
            html.Hr(),
            dbc.Nav([
                dbc.NavLink('Exercise Tracker', href='/', active='exact'),
                dbc.NavLink('Workout Input Form', href='/input', active='exact')
            ],
            vertical=True,
            pills=True)],
            style={'position':'fixed',
                   'top':0,
                   'left':0,
                   'bottom':0,
                   'width':'16rem',
                   'padding':'2rem 1rem'})

content = html.Div(
                id='page_content',
                children=[],
                style={'margin-left':'18rem',
                       'margin-right':'2rem',
                       'padding':'2rem 1rem'})

app.layout = html.Div([
                dcc.Location(id='url'),
                sidebar,
                content])

def exercise_tracker_layout():
    return dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H1("Fitness Analysis Dashboard",
                    className='text-center text-primary, mb-4')
        )
    ),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(id='bodySplit',
                         options=[{'label':x,'value':x}
                                  for x in df['body_split'].unique()],
                         value='Upper',
                         clearable=False,
                         className='mb-4',
                         style={'color': 'black'}),
            width={'size':3}
        ),
        dbc.Col(
            dcc.Dropdown(id='exercise',
                         options=[{'label':x,'value':x}
                                  for x in df['exercise_name'].unique()],
                         value='Y Raises',
                         clearable=False,
                         className='mb-4',
                         style={'color': 'black'}),
            width={'size':3}
        ),
        dbc.Col(
            dcc.Dropdown(id='duration',
                         options=[
                             {'label':'6 Weeks','value':6},
                             {'label':'8 Weeks','value':8},
                             {'label':'10 Weeks','value':10},
                             {'label':'12 Weeks','value':12}],
                             value=6,
                             clearable=False,
                             className='mb-4',
                             style={'color': 'black'}),
            width={'size':3}
        ),
        dbc.Col(
            dcc.Dropdown(id='setNumber',
                         options=[
                             {'label':'1st Set','value':1},
                             {'label':'2nd Set','value':2}],
                             value=1,
                             clearable=False,
                             className='mb-4',
                             style={'color': 'black'}),
            width={'size':3}
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='line_graph',
                      style={"border": "5px solid #dddddd"}),
            width={'size':8}
        ),
        dbc.Col([
            dbc.Card(id='weight_increase',
                     style={"border": "5px solid #dddddd"}),
            dbc.Card(id='consistency',
                     style={"border": "5px solid #dddddd"}),
            dbc.Card(id='1rm',
                     style={"border": "5px solid #dddddd"}),
            dbc.Card(id='1rm_increase',
                     style={"border": "5px solid #dddddd"})],
            width={'size':4}
        )
    ])
])

# App Callback - updating page content based on side bar selection
@app.callback(
    Output(component_id='page_content',component_property='children'),
    Input(component_id='url', component_property='pathname')
)

def update_layout(pathname):
    if pathname == '/':
        return exercise_tracker_layout()
    else:
        return html.H1("Placeholder")
    
# App Callback - connecting inputs/outputs to the dash components
@app.callback(
    [Output(component_id='line_graph', component_property='figure'),
     Output(component_id='weight_increase', component_property='children'),
     Output(component_id='consistency',component_property='children'),
     Output(component_id='1rm_increase',component_property='children'),
     Output(component_id='1rm',component_property='children')],
     [Input(component_id='bodySplit', component_property='value'),
      Input(component_id='exercise',component_property='value'),
      Input(component_id='duration', component_property='value'),
      Input(component_id='setNumber', component_property='value')]
)

def update_dashboard(bodySplitSlct, exerciseSlct, durationSlct, setNumberSlct):

    # Filtering down the dataset into specific exercise
    dff = df.copy()
    dff = dff[dff['body_split'] == bodySplitSlct]
    dff = dff[dff['exercise_name'] == exerciseSlct]
    dff = dff[dff['set_number_exercise'] == setNumberSlct]

    # Filtering down to the specific time period
    dff['date'] = pd.to_datetime(dff['date'])
    dff = dff.sort_values('date')

    end_date = dff['date'].max()

    start_date = end_date - timedelta(weeks = durationSlct)

    dff = dff[(dff['date'] >= start_date) & (dff['date'] <= end_date)]

    # Check if there is any data to show
    if dff.empty:
        return (
            go.Figure().update_layout(
                plot_bgcolor='#303030',
                paper_bgcolor='#222222',
                font=dict(color='white'),
                title={
                        'text':'No Data To Display',
                        'x':0.5,
                        'font':dict(size=30)}),
            dbc.CardBody([
                    html.H4("Weight Increase"),
                    html.H4('No Data')]),
            dbc.CardBody([
                    html.H4("Consistency Score"),
                    html.H4('No Data')]),
            dbc.CardBody([
                    html.H4("Estimated 1RM Increase"),
                    html.H4('No Data')]),
            dbc.CardBody([
                    html.H4("Estimated 1RM"),
                    html.H4('No Data')]),
        )

    # Create a line graph with date along the x-axis, and weight and reps along with 2 y-axes

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=dff['date'], y=dff['weight_kg'],
            mode='lines+markers',
            name='weight_kg',
            line=dict(width=4),
            marker=dict(size=10)),
            secondary_y=False)

    fig.add_trace(
        go.Scatter(
            x=dff['date'], y=dff['reps'],
            mode='lines+markers',
            name='Reps',
            line=dict(width=4),
            marker=dict(size=10)),
            secondary_y=True)

    fig.update_layout(title={
                        'text':'Weight and Reps Over Time',
                        'x':0.5,
                        'font':dict(size=30)},
                      title_x=0.5,
                      plot_bgcolor='#303030',
                      paper_bgcolor='#222222',
                      font=dict(color='#f8f9fa'),
                      legend=dict(
                        font=dict(size=14)))

    fig.update_xaxes(title_text='Time Period of Analysis',
                     title_font=dict(size=18))

    fig.update_yaxes(title_text='Weight (kg)', 
                     title_font=dict(size=18),
                     secondary_y=False)
    
    fig.update_yaxes(title_text='Reps',
                     title_font=dict(size=18),
                     secondary_y=True)

    # Calculating the percentage increase in weight since the start of the analysis
    dff = dff.sort_values('date')

    start_weight = dff['weight_kg'].iloc[0]
    end_weight = dff['weight_kg'].iloc[-1]

    if start_weight > 0:
        pct_increase = ((end_weight - start_weight)/start_weight) * 100
    else:
        pct_increase = 0

    weight_card = dbc.CardBody([
                    html.H4("Weight Increase"),
                    html.H4(f'{pct_increase:.2f} %')]
    )

    # Calculating the consistency score of the time period
    expected_sessions = (durationSlct * 7)/4 # Calculates the number of sessions expected if one rest day is taken between upper and lower sessions

    actual_sessions = len(dff) # Counts the number of rows in dff, and therefore counts how many times I completed the exercise

    consistency_score = (actual_sessions/expected_sessions) * 100

    consistency_card = dbc.CardBody([
                        html.H4("Consistency Score"),
                        html.H4(f'{consistency_score:.1f} %')]
    )

    # Calculating the estimated 1RM increase over the time period using Epley's 1RM formula
    dff = dff.sort_values('date')

    start_weight = dff['weight_kg'].iloc[0]
    end_weight = dff['weight_kg'].iloc[-1]

    start_reps = dff['reps'].iloc[0]
    end_reps = dff['reps'].iloc[-1]
    
    start_1rm = start_weight * (1 + (start_reps/30))
    end_1rm = end_weight * (1 + (end_reps/30))

    increase_score = end_1rm - start_1rm

    increase_card = dbc.CardBody([
                        html.H4("Estimated 1RM Increase"),
                        html.H4(f'{increase_score:.2f} kg')]
    )

    one_rep_max = dbc.CardBody([
                        html.H4("Estimated 1RM"),
                        html.H4(f'{end_1rm:.2f} kg')
    ])

    return fig, weight_card, consistency_card, increase_card, one_rep_max



if __name__ == '__main__':
    app.run_server(debug=True)