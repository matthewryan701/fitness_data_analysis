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

# Creating the app with DARKLY's bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

# Importing the data
engine = create_engine('postgresql://postgres:D4t4B4s3@localhost:5432/fitness_database')
df = pd.read_sql('SELECT * FROM exercises', engine)
df['date'] = pd.to_datetime(df['date'])

# Using csv to create lookup table
lookup_table = pd.read_csv('exercise_lookup_table.csv')
lookup_dict = lookup_table.set_index('exercise_name')[['body_split', 'muscle_group']].to_dict('index')

# Navigation bar down the left side of the website
sidebar = html.Div([
            html.H2('Sidebar'),
            html.Hr(),
            dbc.Nav([
                dbc.NavLink('Exercise Tracker', href='/', active='exact'),
                dbc.NavLink('Workout Input Form', href='/input', active='exact'),
                dbc.NavLink('Body Weight Tracker', href='/bodyweight', active='exact')
            ],
            vertical=True,
            pills=True)],
            style={'position':'fixed',
                   'top':0,
                   'left':0,
                   'bottom':0,
                   'width':'16rem',
                   'padding':'2rem 1rem'})

# HTML Div container to store page contents
content = html.Div(
                id='page_content',
                children=[],
                style={'margin-left':'18rem',
                       'margin-right':'2rem',
                       'padding':'2rem 1rem'})

# General app layout - nav bar on the left, page contents on the right
app.layout = html.Div([
                dcc.Location(id='url'),
                sidebar,
                content])

# Function to return the exercise tracker's layout
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
    ])])

# Function to return the input form's layout
def input_form_layout():
    return dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H1("Workout Data Input Form",
                    className='text-center text-primary, mb-4')
        )
    ),
    dbc.Row(
        dbc.Col(
            dbc.Alert(id='null_error', 
                      color='primary', 
                      is_open=False,
                      className='text-center text-primary, mb-4')
            )
    ),
    dbc.Row(
        dbc.Col(
            dcc.Slider(id='set_slider',
                       min=5,
                       max=20,
                       step=1,
                       value=5,
                       className='mb-4')
        )
    ),
    dbc.Row(
        dbc.Col(
            html.Div(id='input_boxes')
        )
    ),
    dbc.Row(
        dbc.Col(
            html.Div(
                dbc.Button("Submit Workout",
                            id='submit_workout'),
                className='d-grid col-6 mx-auto'   
            )
        )
    )])

# App Callback - updating page content based on side bar selection
@app.callback(
    Output(component_id='page_content',component_property='children'),
    Input(component_id='url', component_property='pathname')
)

def update_layout(pathname):
    if pathname == '/':
        return exercise_tracker_layout()
    elif pathname == '/input':
        return input_form_layout()
    else:
        return bodyweight_tracker_layout()
    
# App Callback - connecting the dashboard drop downs to the graphs and stat cards
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

# App Callback - change number of boxes on input form with slider
@app.callback(
    Output(component_id='input_boxes', component_property='children'),
    Input(component_id='set_slider', component_property='value')
)

def update_boxes(n_sets):
    
    date_inputs = []
    exercise_inputs = []
    weight_inputs = []
    reps_inputs = []

    for i in range(n_sets):
        date_inputs.append(
            dbc.Input(id={'type': 'date', 'index': i}, type='date', placeholder='Date', required=True, className='mb-2')
        )
        exercise_inputs.append(
            dcc.Dropdown(id={'type': 'exercise', 'index': i},
                         options=[{'label':x,'value':x}
                                  for x in df['exercise_name'].unique()],
                         placeholder='Exercise',
                         clearable=False,
                         className='mb-2',
                         style={'color': 'black'}),
        )
        weight_inputs.append(
            dbc.Input(id={'type': 'weight', 'index': i}, type='number', placeholder='Weight (kg)', required=True, className='mb-2',
                      min=0, max=500)
        )
        reps_inputs.append(
            dbc.Input(id={'type': 'reps', 'index': i}, type='number', placeholder='Reps', required=True, className='mb-2',
                      min=1, max=50)
        )
    
    return dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4("Date"),
                    *date_inputs
                ]),
                style={"border": "5px solid #dddddd"},
                className='mb-4'
            ),
            width={'size':3}
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4("Exercise"),
                    *exercise_inputs
                ]),
                style={"border": "5px solid #dddddd"},
                className='mb-4'
            ),
            width={'size':3}
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4("Weight"),
                    *weight_inputs
                ]),
                style={"border": "5px solid #1f77b4"},
                className='mb-4'
            ),
            width={'size':3}
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4("Reps"),
                    *reps_inputs
                ]),
                style={"border": "5px solid #ff7f0e"},
                className='mb-4'
            ),
            width={'size':3}
        )
    ])

# App Callback - store input form data into SQL database with submit button
@app.callback(
    [Output(component_id='null_error', component_property='children'),
    Output(component_id='null_error', component_property='is_open'),
    Output({'type': 'date', 'index': ALL}, 'value'),
    Output({'type': 'exercise', 'index': ALL}, 'value'),
    Output({'type': 'weight', 'index': ALL}, 'value'),
    Output({'type': 'reps', 'index': ALL}, 'value')],
    Input(component_id='submit_workout', component_property='n_clicks'),
    State(component_id={'type': 'date', 'index': ALL}, component_property='value'),
    State(component_id={'type': 'exercise', 'index': ALL}, component_property='value'),
    State(component_id={'type': 'weight', 'index': ALL}, component_property='value'),
    State(component_id={'type': 'reps', 'index': ALL}, component_property='value')
)

def on_submit(n_clicks, dates, exercises, weights, reps):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    null_error = False 
    # Handling missing values
    for d, e, w, r in zip(dates, exercises, weights, reps):
        if not all ([d, e, w, r]):
            null_error = True
            return ('Please fill all boxes on the screen with data.', 
                    True,
                    [dash.no_update] * len(dates), 
                    [dash.no_update] * len(exercises), 
                    [dash.no_update] * len(weights), 
                    [dash.no_update] * len(reps))
    
    # Looking up body split and muscle group from the lookup dictionary
    split = [lookup_dict.get(e, {}).get('body_split', 'Unknown') for e in exercises]
    muscle = [lookup_dict.get(e, {}).get('muscle_group', 'Unknown') for e in exercises]

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    table = {'date':dates,'exercise_name':exercises,'body_split':split,'muscle_group':muscle,'weight_kg':weights,
             'reps':reps}
    
    dff = pd.DataFrame(table)

    # Calculating ordinal fields for the data
    dff['exercise_number'] = dff.groupby('date')['exercise_name'].transform(lambda x: pd.factorize(x)[0]+1)

    dff['set_number_exercise'] = dff.groupby(['date','exercise_name']).cumcount()+1

    dff['set_number_session'] = dff.groupby('date').cumcount()+1

    try:
        dff.to_sql('exercises', engine, if_exists='append', index=False)
    except SQLAlchemyError as e:
        return f"Database error: {str(e)}", True, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    num_fields = len(dates)
    return "Data successfully stored.", True, [None]*num_fields, [None]*num_fields, [None]*num_fields, [None]*num_fields

if __name__ == '__main__':
    app.run_server(debug=True)