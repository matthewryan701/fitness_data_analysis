import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State, ALL
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import timedelta, datetime

# Creating the app with DARKLY's bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions = True)

# Importing the data
engine = create_engine('postgresql://postgres:D4t4B4s3@localhost:5432/fitness_database')
# Exercises Table
df = pd.read_sql('SELECT * FROM exercises', engine)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
# Exercise Lookup Table
exercise_lookup = pd.read_sql('SELECT * FROM exercise_lookup', engine)
lookup_dict = exercise_lookup.set_index('exercise_name')[['body_split', 'muscle_group']].to_dict('index')
# Body Weight Table
dfbody = pd.read_sql('SELECT * FROM body_weight', engine)
dfbody['date'] = pd.to_datetime(dfbody['date'])
dfbody = dfbody.sort_values('date')
# Cardio Table
dfcardio = pd.read_sql('SELECT * FROM cardio', engine)
dfcardio['date'] = pd.to_datetime(dfcardio['date'])
dfcardio = dfcardio.sort_values('date')
# Cardio Lookup Table
dfcardiolookup = pd.read_sql('SELECT * FROM cardio_lookup', engine)

# Navigation bar down the left side of the website
sidebar = html.Div([
            html.Img(src='/assets/fort_analytics.jpg',
                     style={'width':'100%',
                            'maxHeight':'200px',
                            'objectFit':'contain',
                            'marginBottom':'1rem'}),
            html.Hr(),
            dbc.Nav([
                dbc.NavLink('Home Page', href='/', active='exact'),
                dbc.NavLink('Exercise Analytics', href='/exercise', active='exact'),
                dbc.NavLink('Workout Input Form', href='/input', active='exact'),
                dbc.NavLink('Body Weight Tracker', href='/bodyweight', active='exact'),
                dbc.NavLink('Cardio Tracker', href='/cardio', active='exact')
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

# Home Page layout
def home_page_layout():
    # Calculating average calorie rate for last 2 weeks based on BMR 
    recent_two_weeks = datetime.now() - timedelta(weeks=2)
    recent_dfbody = dfbody[dfbody['date'] >= recent_two_weeks]
    avg_bw = recent_dfbody['body_weight'].mean()

    bmr = (9.65*avg_bw)+(573*1.98)-(5.08*19)+260
    kcal_per_day = bmr*1.5

    grams_fat = (kcal_per_day*0.4)/8
    grams_carbs = (kcal_per_day*0.35)/4
    grams_protein = (kcal_per_day*0.25)/4

    pie = go.Figure(
        data=[go.Pie(
            labels=[f'{grams_fat:.1f}g Fat',
                    f'{grams_carbs:.1f}g Carbs',
                    f'{grams_protein:.1f}g Protein'],
            values=[0.4, 0.35, 0.25],
            textposition='outside',
            textinfo='label+text', 
            textfont=dict(size=16,
                          family='Roboto, sans-serif'),
            showlegend=False,
            pull=[0.05, 0.05, 0.05],
            marker=dict(line=dict(color='#f8f9fa', width=1)),
        )]
    )
    pie.update_layout(
        title={'text':f'{kcal_per_day:.0f} Calories',
               'font':dict(size=30,
                           family='Roboto, sans-serif')},
               plot_bgcolor='#303030',
               paper_bgcolor='#222222',
               font=dict(color='#f8f9fa'),
               showlegend=False,
               title_x=0.5)

    # Creating calendar heatmap
    today = datetime.today()
    start_of_month = datetime(today.year, today.month, 1)
    end_of_month = (start_of_month + pd.offsets.MonthEnd()).date()

    calendar_days = pd.date_range(start=start_of_month, end=end_of_month)

    calendar_df = pd.DataFrame({
        'date':calendar_days,
        'day':calendar_days.day,
        'weekday':calendar_days.weekday,
        'week':calendar_days.to_series().apply(lambda d: (d.day - 1 + start_of_month.weekday()) // 7)
    })

    lifting_dates = df['date'].dt.normalize().unique()

    calendar_df['lifted'] = calendar_df['date'].isin(lifting_dates).astype(int)

    split_by_date = df.groupby('date')['body_split'].first()

    heatmap_z = []
    heatmap_text = []
    max_week = calendar_df['week'].max()

    for week in range(max_week + 1):
        row = []
        text_row = []
        for day in range(7):  # Monday to Sunday
            day_match = calendar_df[(calendar_df['week'] == week) & (calendar_df['weekday'] == day)]
            if not day_match.empty:
                date_val = day_match['date'].values[0]
                lifted = day_match['lifted'].values[0]
                day_num = day_match['day'].values[0]

                split = split_by_date.get(pd.to_datetime(date_val), None)
                hover = f"{day_num}<br>{split}" if lifted else str(day_num)

                row.append(lifted)
                text_row.append(hover)
            else:
                row.append(None)  # Fill empty cells with None
                text_row.append("")
        heatmap_z.append(row)
        heatmap_text.append(text_row)

    heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_z,
        text=heatmap_text,
        hoverinfo='text',
        colorscale=[[0, '#303030'], [1, '#3D9970']],  
        showscale=False,
        x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        y=[f"Week {i+1}" for i in range(len(heatmap_z))],
        texttemplate="%{text}",
        textfont={"size":14}
    ))
    heatmap.update_layout(
        title={'text':f'Dates Lifted {today.strftime('%B %Y')}',
               'font':dict(size=30,
                           family='Roboto, sans-serif')},
        xaxis=dict(showgrid=False, side='top'),
        yaxis=dict(showgrid=False, autorange='reversed'),
        margin=dict(t=60, l=40, r=40, b=40),
        height=300 + 40 * len(heatmap_z),
        title_x=0.5,
        plot_bgcolor='#303030',
        paper_bgcolor='#222222',
        font=dict(color='#f8f9fa',
                  family='Roboto, sans-serif')
    )
    heatmap.update_xaxes(title_font=dict(size=18,
                                         family='Roboto, sans-serif'))
    heatmap.update_yaxes(title_font=dict(size=18,
                                         family='Roboto, sans-serif'))

    # Creating consistency bar chart over last 30 days
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    today = datetime.today().date()
    start_date = pd.to_datetime(today - timedelta(days=30))

    recent_df = df[df['date'] >= start_date]

    period_days = 30
    ideal_frequency = period_days / 4  

    exercise_dates = (
        recent_df.groupby('exercise_name')['date']
        .nunique()  
        .reset_index()
        .rename(columns={'date': 'actual_count'})
    )

    # Calculate consistency as percentage
    exercise_dates['consistency_score'] = (
        exercise_dates['actual_count'] / ideal_frequency * 100
    ).clip(upper=100).round(1) 
    exercise_dates_sorted = exercise_dates.sort_values(by='consistency_score', ascending=False)
    exercise_dates_sorted['exercise_id'] = [f'#{i+1}' for i in range(len(exercise_dates_sorted))]

    bar = px.bar(
        exercise_dates_sorted,
        x='exercise_id',
        y='consistency_score',
        title='Consistency Scores Over Last Month',
        labels={'consistency_score': 'Consistency (%)', 'exercise_name': 'Exercise'},
        hover_name='exercise_name',
        range_y=[0, 100],
        color_discrete_sequence=['#007BFF']
    )
    bar.update_layout(
        title={'text':'Consistency Scores',
               'font':dict(size=30,
                           family='Roboto, sans-serif')},
        annotations=[dict(text="Last 30 Days",
                          x=0.5,
                          y=1.2,  
                          xref="paper",
                          yref="paper",
                          showarrow=False,
                          font=dict(size=16, color="gray", family='Roboto, sans-serif'),
                          xanchor='center')],
        yaxis_title='Consistency (%)',
        xaxis_title='Exercise',
        title_x=0.5,
        plot_bgcolor='#303030',
        paper_bgcolor='#222222',
        font=dict(color='#f8f9fa')
    )

    # Calculate best and worst exercises in last 30 days by % increase in weight
    recent_df = recent_df.sort_values('date')
    recent_df['one_rm'] = recent_df['weight_kg'] * (1 + (recent_df['reps']/30))
    one_rm_progress = recent_df.groupby('exercise_name')['one_rm'].agg(
        start_1rm = 'first',
        end_1rm = 'last').reset_index()
    one_rm_progress['pct_increase'] = (
        (one_rm_progress['end_1rm'] - one_rm_progress['start_1rm'])/
        one_rm_progress['start_1rm']) * 100

    best_exercise = one_rm_progress.loc[one_rm_progress['pct_increase'].idxmax()]

    worst_exercise = one_rm_progress.loc[one_rm_progress['pct_increase'].idxmin()]

    # Progress circle to show bodyweight tracking in last 7 days
    today = datetime.now().date()
    one_week_ago = today - timedelta(days=6)  

    recent_bw_logs = dfbody[dfbody['date'].dt.date >= one_week_ago]
    logged_days = recent_bw_logs['date'].dt.date.nunique()

    progress = logged_days / 7 

    bw_progress = go.Figure(go.Pie(
        values=[progress, 1 - progress],
        hole=0.8,
        marker_colors=["#39FF14", '#303030'],  
        textinfo='none',
        sort=False
    ))

    bw_progress.update_layout(
        title={'text':'Body Weight Tracking',
               'font':dict(size=30,
                           family='Roboto, sans-serif')},
        annotations=[
        dict(text="Last 7 Days",
             x=0.5,
             y=1.2,  
             xref="paper",
             yref="paper",
             showarrow=False,
             font=dict(size=16, color="gray", family='Roboto, sans-serif'),
             xanchor='center'),
        dict(text=f"{logged_days}/7",
             font=dict(size=30, family='Roboto, sans-serif'),
             showarrow=False)],
        showlegend=False,
        title_x=0.5,
        plot_bgcolor='#303030',
        paper_bgcolor='#222222',
        font=dict(color='#f8f9fa', family='Roboto, sans-serif'))
    
    return dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H1("Overall Fitness Summary",
                    className='text-center text-primary, mb-3')
        )
    ),
    html.Hr(),
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='calorie_pie',
                      figure=pie,
                      style={'height':'300px'}),
            width={'size':4}
        ),
        dbc.Col(
            dcc.Graph(id='calendar_heatmap',
                      figure=heatmap,
                      style={'height':'300px'}),
            width={'size':8}
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='consistency_bar',
                      figure=bar,
                      style={'height':'360px'}),
            width={'size':6}
        ),
        dbc.Col([
            dbc.Card(
                dbc.CardBody([html.H4("Best Exercise",
                                      className='text-center text-primary, mb-2'),
                              html.Hr(className="bg-secondary"),
                              html.H5(f'{best_exercise['exercise_name']}',
                                      className='text-center text-light, mb-2')]),
                style={'height':'165px',
                       "border": "3px solid #28a745"},
                className="rounded-3 shadow"),
            html.Hr(),
            dbc.Card(
                dbc.CardBody([html.H4("Worst Exercise",
                                      className='text-center text-primary, mb-2'),
                              html.Hr(className="bg-secondary"),
                              html.H5(f'{worst_exercise['exercise_name']}',
                                      className='text-center text-light, mb-2')]),
                style={'height':'165px',
                       'border':'3px solid #dc3545'},
                className="rounded-3 shadow")],
            width={'size':2}
        ),
        dbc.Col(
            dcc.Graph(id='bw_progress_circle',
                      figure=bw_progress,
                      style={'height':'360px'}),
            width={'size':4}
        )
    ])], fluid=True, className="p-4")

# Exercise Analytics layout
def exercise_tracker_layout():
    return dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H1("Exercise Analytics",
                    className='text-center text-primary, mb-3')
        )
    ),
    html.Hr(),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(id='bodySplit',
                         options=[{'label':x,'value':x}
                                  for x in exercise_lookup['body_split'].unique()],
                         value='Upper',
                         clearable=False,
                         className='mb-4',
                         style={'color': 'black'}),
            width={'size':3}
        ),
        dbc.Col(
            dcc.Dropdown(id='exercise',
                         options=[{'label':x,'value':x}
                                  for x in exercise_lookup['exercise_name'].unique()],
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
                      style={'height':'500px'}),
            width={'size':8}
        ),
        dbc.Col([
            dbc.Card(id='weight_increase'),
            html.Hr(),
            dbc.Card(id='consistency'),
            html.Hr(),
            dbc.Card(id='1rm'),
            html.Hr(),
            dbc.Card(id='1rm_increase')],
            width={'size':4}
        )
    ])], fluid=True, className="p-4")

# Input Form layout
def input_form_layout():
    return dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H1("Workout Input Form",
                    className='text-center text-primary, mb-3')
        )
    ),
    html.Hr(),
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
                            id='submit_workout',
                            style={'backgroundColor':"#3D9970"}),
                className='d-grid col-6 mx-auto'   
            )
        )
    )], fluid=True, className="p-4")

# Body Weight Tracker layout
def bodyweight_tracker_layout():
    bodyweight_graph = go.Figure()
    bodyweight_graph.add_trace(go.Scatter(x=dfbody['date'],
                                          y=dfbody['body_weight'],
                                          mode='lines+markers',
                                          name='body_weight',
                                          line=dict(width=4, color='#39FF14'),
                                          marker=dict(size=10, color='#39FF14')))
    bodyweight_graph.update_layout(title={'text':'Body Weight Over Time',
                                          'font':dict(size=30, family='Roboto, sans-serif')},
                                    annotations=[dict(text="Readings Taken In The Morning Before Consuming Anything",
                                                      x=0.5,
                                                      y=1.16,  
                                                      xref="paper",
                                                      yref="paper",
                                                      showarrow=False,
                                                      font=dict(size=16, color="gray", family='Roboto, sans-serif'),
                                                      xanchor='center')],
                                    xaxis_title='Date',
                                    yaxis_title='Body Weight (kg)',
                                    title_x=0.5,
                                    plot_bgcolor='#303030',
                                    paper_bgcolor='#222222',
                                    font=dict(color='#f8f9fa', family='Roboto, sans-serif'))
    bodyweight_graph.update_xaxes(title_text='Date',
                                  title_font=dict(size=18, family='Roboto, sans-serif'))
    bodyweight_graph.update_yaxes(title_text='Body Weight (kg)',
                                  title_font=dict(size=18, family='Roboto, sans-serif'))
    return dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H1("Body Weight Tracker",
                    className='text-center text-primary, mb-3')
        )
    ),
    html.Hr(),
    dbc.Row(
        dbc.Col(
            dbc.Alert(id='bodyweight_null',
                      color='primary',
                      is_open=False,
                      className='text-center text-primary, mb-4')
        )
    ),
    dbc.Row([
        dbc.Col(
            dbc.Input(id='bodyweight_date',
                      type='date',
                      placeholder='Date',
                      required=True,
                      className='mb-3',),
            width={'size':4}
        ),
        dbc.Col(
            dbc.Input(id='bodyweight_input',
                      type='number',
                      placeholder='Body Weight (kg)',
                      required=True,
                      className='mb-3',
                      style={'height':'40px'}),
            width={'size':4}
        ),
        dbc.Col(
            html.Div(
                dbc.Button("Submit Body Weight",
                           id='submit_bodyweight',
                           style={"backgroundColor": "#39FF14",
                                  'color':'black',
                                  'height':'40px'},
                           className='mb-3'),
                           className='d-grid mx-auto'),
            width={'size':4}
        )
    ]),
    dbc.Row(
        dbc.Col(
            dcc.Graph(id='bodyweight_graph',
                      figure=bodyweight_graph)
        )
    )], fluid=True, className="p-4")

# Cardio Tracker Layout
def cardio_tracker_layout():
    return dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H1("Cardio Tracker",
                    className='text-center text-primary, mb-3')
        )
    ),
    html.Hr(),
    dbc.Row(
        dbc.Col(
            dbc.Alert(id='cardio_null',
                      color='primary',
                      is_open=False,
                      className='text-center text-primary, mb-4')
        )
    ),
    dbc.Row([
        dbc.Col(
            dbc.Input(id='cardio_date',
                      type='date',
                      placeholder='Date',
                      required=True,
                      style={'height':'40px',
                             'display':'flex',
                             'alignItems':'center',
                             'justifyContent':'center'},
                      className='mb-3'),
            width={'size':2}
        ),
        dbc.Col(
            dcc.Dropdown(id='cardio_selection',
                         options=[{'label':x,'value':x}
                                  for x in dfcardiolookup['cardio_type'].unique()],
                                  value='Walking',
                                  clearable=False,
                                  className='mb-3',
                                  style={'color':'black',
                                         'height':'40px'}),
            width={'size':2}
        ),
        dbc.Col(
            dcc.Dropdown(id='cardio_intensity',
                         options=[{'label':x,'value':x}
                                  for x in dfcardiolookup['intensity'].unique()],
                                  value='Light',
                                  clearable=False,
                                  className='mb-3',
                                  style={'color':'black',
                                         'height':'40px'}),
            width={'size':2}
        ),
        dbc.Col(
            dbc.Input(id='cardio_duration',
                      type='number',
                      placeholder='Duration (mins)',
                      required=True,
                      style={'height':'40px',
                             'display':'flex',
                             'alignItems':'center',
                             'justifyContent':'center'},
                      className='mb-3'),
            width={'size':2}
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    html.H5(id='cardio_calories'),
                    style={'display':'flex',
                           'alignItems':'center',
                           'justifyContent':'center',
                           'height':'100%'}
                ),
                style={'height':'40px'},
                className='mb-3'
            ),
            width={'size':2}
        ),
        dbc.Col(
            dbc.Button('Submit Cardio',
                       id='submit_cardio',
                       style={'height':'40px',
                              'display':'flex',
                              'alignItems':'center',
                              'justifyContent':'center',
                              'backgroundColor':'#3D9970'},
                        className='mb-3'),
            width={'size':2}
        )
    ]),
    dbc.Row(
        dbc.Col(
            dcc.Slider(id='cardio_slider',
                       min=4,
                       max=20,
                       step=1,
                       value=4,
                       className='mb-4')
        )
    ),
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='cardio_graph',
                      style={'height':'400px'}),
            width={'size':8}
        ),
        dbc.Col([
            dbc.Card(id='most_frequent_cardio',
                     style={'height':'180px'}),
            html.Hr(),
            dbc.Card(id='average_calories_burned',
                     style={'height':'180px'})],
            width={'size':4})
    ])
    ], fluid=True, className="p-4")

# ----- General Layout -----
@app.callback(
    Output(component_id='page_content',component_property='children'),
    Input(component_id='url', component_property='pathname')
)

def update_layout(pathname):
    if pathname == '/':
        return home_page_layout()
    elif pathname == '/exercise':
        return exercise_tracker_layout()
    elif pathname == '/input':
        return input_form_layout()
    elif pathname == '/bodyweight':
        return bodyweight_tracker_layout()
    else:
        return cardio_tracker_layout()
    
# ----- Exercise Analytics -----
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
                font=dict(color='white', family='Roboto, sans-serif'),
                title={
                        'text':'No Data To Display',
                        'x':0.5,
                        'font':dict(size=30, family='Roboto, sans-serif')}),
            dbc.CardBody([
                    html.H4("Weight Increase", className='text-center text-primary, mb-2'),
                    html.H4('No Data', className='text-center text-primary, mb-2')]),
            dbc.CardBody([
                    html.H4("Consistency Score", className='text-center text-primary, mb-2'),
                    html.H4('No Data', className='text-center text-primary, mb-2')]),
            dbc.CardBody([
                    html.H4("Estimated 1RM Increase", className='text-center text-primary, mb-2'),
                    html.H4('No Data', className='text-center text-primary, mb-2')]),
            dbc.CardBody([
                    html.H4("Estimated 1RM", className='text-center text-primary, mb-2'),
                    html.H4('No Data', className='text-center text-primary, mb-2')]),
        )

    # Create a line graph with date along the x-axis, and weight and reps along with 2 y-axes

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=dff['date'], y=dff['weight_kg'],
            mode='lines+markers',
            name='Weight (kg)',
            line=dict(width=4),
            marker=dict(size=10,
                        color='#ef553b')),
            secondary_y=False)

    fig.add_trace(
        go.Scatter(
            x=dff['date'], y=dff['reps'],
            mode='lines+markers',
            name='Reps',
            line=dict(width=4),
            marker=dict(size=10,
                        color="#636efa")),
            secondary_y=True)

    fig.update_layout(title={
                        'text':f'Weight and Reps',
                        'font':dict(size=30, family='Roboto, sans-serif')},
                        annotations=[
                            dict(text=f"Last {durationSlct} Weeks",
                                 x=0.57,
                                 y=1.12,  
                                 xref="paper",
                                 yref="paper",
                                 showarrow=False,
                                 font=dict(size=16, color="gray", family='Roboto, sans-serif'),
                                 xanchor='center')],
                        yaxis2=dict(tickmode='linear',
                                    dtick=1,            
                                    tickformat='d'),
                      title_x=0.5,
                      plot_bgcolor='#303030',
                      paper_bgcolor='#222222',
                      font=dict(color='#f8f9fa', family='Roboto, sans-serif'),
                      legend=dict(font=dict(size=14, family='Roboto, sans-serif')))

    fig.update_xaxes(title_text='Time Period of Analysis',
                     title_font=dict(size=18, family='Roboto, sans-serif'))

    fig.update_yaxes(title_text='Weight (kg)', 
                     title_font=dict(size=18, family='Roboto, sans-serif'),
                     secondary_y=False)
    
    fig.update_yaxes(title_text='Reps',
                     title_font=dict(size=18, family='Roboto, sans-serif'),
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
                    html.H4("Weight Increase", className='text-center text-primary, mb-2'),
                    html.H4(f'{pct_increase:.2f} %', className='text-center text-primary, mb-2')]
    )

    # Calculating the consistency score of the time period
    expected_sessions = (durationSlct * 7)/4 # Calculates the number of sessions expected if one rest day is taken between upper and lower sessions

    actual_sessions = len(dff) # Counts the number of rows in dff, and therefore counts how many times I completed the exercise

    consistency_score = (actual_sessions/expected_sessions) * 100

    consistency_card = dbc.CardBody([
                        html.H4("Consistency Score", className='text-center text-primary, mb-2'),
                        html.H4(f'{consistency_score:.1f} %', className='text-center text-primary, mb-2')]
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
                        html.H4("Estimated 1RM Increase", className='text-center text-primary, mb-2'),
                        html.H4(f'{increase_score:.2f} kg', className='text-center text-primary, mb-2')]
    )

    one_rep_max = dbc.CardBody([
                        html.H4("Estimated 1RM", className='text-center text-primary, mb-2'),
                        html.H4(f'{end_1rm:.2f} kg', className='text-center text-primary, mb-2')
    ])

    return fig, weight_card, consistency_card, increase_card, one_rep_max

# ----- Input Form -----
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
                                  for x in exercise_lookup['exercise_name'].unique()],
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
                style={"border": "5px solid #ef553b"},
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
                style={"border": "5px solid #636efa"},
                className='mb-4'
            ),
            width={'size':3}
        )
    ])

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

# ----- Body Weight Tracker -----
@app.callback(
    [Output(component_id='bodyweight_null', component_property='children'),
    Output(component_id='bodyweight_null', component_property='is_open'),
    Output(component_id='bodyweight_date', component_property='value'),
    Output(component_id='bodyweight_input', component_property='value')],
    Input(component_id='submit_bodyweight', component_property='n_clicks'),
    State(component_id='bodyweight_date', component_property='value'),
    State(component_id='bodyweight_input', component_property='value')
)

def store_bodyweight(n_clicks, date, bodyweight):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    bodyweight_null = False
    # Handling missing values
    if not date or not bodyweight:
        bodyweight_null = True
        return ('Please fill the input boxes with data.',
                True,
                dash.no_update,
                dash.no_update)
        
    table = [{'date':date,
             'body_weight':bodyweight}]
    
    dff = pd.DataFrame(table)

    try:
        dff.to_sql('body_weight', engine, if_exists='append', index=False)
    except SQLAlchemyError as e:
        return f"Database error: {str(e)}", True, dash.no_update, dash.no_update
    
    return "Data successfully stored.", True, None, None

# ----- Cardio Tracker -----
@app.callback(
    Output(component_id='cardio_calories', component_property='children'),
    [Input(component_id='cardio_selection', component_property='value'),
     Input(component_id='cardio_intensity', component_property='value'),
     Input(component_id='cardio_duration', component_property='value')]
)

def total_cardio_calories(cardio_type, intensity, duration):
    if duration is None:
        return '0 kcal'
    dff = dfcardiolookup.copy()
    dff = dff[dff['cardio_type'] == cardio_type]
    dff = dff[dff['intensity'] == intensity]
    total_calories = (dff['calories_per_min'].iloc[0])*duration
    return f'{round(total_calories)} kcal'

@app.callback(
    [Output(component_id='cardio_null', component_property='children'),
     Output(component_id='cardio_null', component_property='is_open'),
     Output(component_id='cardio_date', component_property='value'),
     Output(component_id='cardio_duration', component_property='value')],
     Input(component_id='submit_cardio', component_property='n_clicks'),
     State(component_id='cardio_date', component_property='value'),
     State(component_id='cardio_selection', component_property='value'),
     State(component_id='cardio_intensity', component_property='value'),
     State(component_id='cardio_duration', component_property='value'),
     State(component_id='cardio_calories', component_property='children')
)

def store_cardio(n_clicks, cardio_date, cardio_slct, cardio_int, cardio_dur, cardio_kcals):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    # Handling missing values
    if not cardio_slct or not cardio_int or not cardio_dur or not cardio_kcals or not cardio_date:
        return ('Please fill the input boxes with data',
                True,
                dash.no_update,
                dash.no_update)
    
    calorie_number = int(cardio_kcals.split()[0])
    table = [{'date':cardio_date,
             'cardio_type':cardio_slct,
             'intensity':cardio_int,
             'duration':cardio_dur,
             'calories_burned':calorie_number}]
    
    dff = pd.DataFrame(table)

    try:
        dff.to_sql('cardio', engine, if_exists='append', index=False)
    except SQLAlchemyError as e:
        return f'Database error: {str(e)}', True, dash.no_update, dash.no_update
    
    return 'Data successfully stored.', True, None, None

@app.callback(
    [Output(component_id='cardio_graph', component_property='figure'),
     Output(component_id='most_frequent_cardio', component_property='children'),
     Output(component_id='average_calories_burned', component_property='children')],
    Input(component_id='cardio_slider', component_property='value')
)

def cardio_graph_time(time_weeks):
    dff = dfcardio.copy()
    end_date = dfcardio['date'].max()
    start_date = end_date - timedelta(weeks=time_weeks)
    dff = dff[(dff['date'] >= start_date) & (dff['date'] <= end_date)]

    if dff.empty:
        return (
            go.Figure().update_layout(
                plot_bgcolor='#303030',
                paper_bgcolor='#222222',
                font=dict(color='white', family='Roboto, sans-serif'),
                title={
                    'text':'No Data To Display',
                    'x':0.5,
                    'font':dict(size=30, family='Roboto, sans-serif')}),
            dbc.CardBody([
                        html.H4("Most Frequent Cardio",
                                className='text-center text-primary, mb-2'),
                        html.H4('No Data',
                                className='text-center text-primary, mb-2')]),
            dbc.CardBody([
                        html.H4("Average Calories Burned",
                                className='text-center text-primary, mb-2'),
                        html.H4('No Data',
                                className='text-center text-primary, mb-2')]))
    
    # Graph
    grouped = dff.groupby('date').agg({
        'calories_burned': 'sum'
    }).reset_index()

    session_details = (
        dff.groupby('date', group_keys=False)
        .apply(lambda g: '<br>'.join(
            f"{row['cardio_type']} | {row['intensity']} | {row['duration']} mins"
            for _, row in g.iterrows()
        )).reset_index(name='session_info')
    )

    # 3. Merge session info into grouped dataframe
    grouped = grouped.merge(session_details, on='date')

    cardio_graph = go.Figure()
    cardio_graph.add_trace(go.Scatter(x=grouped['date'],
                                      y=grouped['calories_burned'],
                                      mode='lines+markers',
                                      line=dict(width=4),
                                      marker=dict(size=10),
                                      customdata=grouped['session_info'],
                                      hovertemplate=
                                       "Date: %{x}<br>" +
                                       "Total Calories: %{y}<br><br>" +
                                       "Sessions:<br>%{customdata}<extra></extra>"))
    
    cardio_graph.update_layout(title={'text':'Calories Burned',
                                      'x':0.5,
                                      'font':dict(size=30, family='Roboto, sans-serif')},
                                annotations=[dict(text="Estimated Values - Per Cardio Session",
                                                  x=0.5,
                                                  y=1.2,  
                                                  xref="paper",
                                                  yref="paper",
                                                  showarrow=False,
                                                  font=dict(size=16, color="gray", family='Roboto, sans-serif'),
                                                  xanchor='center')],
                                title_x=0.5,
                                plot_bgcolor='#303030',
                                paper_bgcolor='#222222',
                                font=dict(color='#f8f9fa', family='Roboto, sans-serif'),
                                legend=dict(font=dict(size=14)))
    
    cardio_graph.update_xaxes(title_text='Time Period of Analysis',
                              title_font=dict(size=18, family='Roboto, sans-serif'))
    
    cardio_graph.update_yaxes(title_text='Calories Burned',
                              title_font=dict(size=18, family='Roboto, sans-serif'))
    
    # Stat cards
    frequent = dff['cardio_type'].mode()[0]
    average = dff['calories_burned'].mean()

    frequency_card = dbc.CardBody([
                        html.H3("Most Frequent Cardio:",
                                className='text-center text-primary, mb-2'),
                        html.H4(f'{frequent}',
                                className='text-center text-primary, mb-2')])
    
    average_card = dbc.CardBody([
                        html.H3("Average Calories Burned:",
                                className='text-center text-primary, mb-2'),
                        html.H4(f'{average:.0f}',
                                className='text-center text-primary, mb-2')])
    
    return cardio_graph, frequency_card, average_card

if __name__ == '__main__':
    app.run_server(debug=True)