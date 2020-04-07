import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import json
import os
import pandas as pd
from datetime import datetime
from datetime import timedelta
from urllib import parse
import requests


logger = logging.getLogger(__name__)

external_stylesheets = [dbc.themes.DARKLY]

is_cf_instance = os.environ.get('CF_INSTANCE_GUID', '') != ''
port = int(os.environ.get('PORT', 8050))
host = os.environ.get('CF_INSTANCE_INTERNAL_IP', '127.0.0.1')

wml_api_key = os.environ['WML_API_KEY']
wml_scoring_url = os.environ['WML_SCORING_URL']
url = parse.urlparse(wml_scoring_url)
wml_base_url = url._replace(path='').geturl()
wml_instance_id = url.path.split('/')[3]

logger.setLevel(logging.INFO if is_cf_instance else logging.DEBUG)

logger.info('Starting %s server: %s:%d', 'CF' if is_cf_instance else 'local', host, port)

logger.info('WML URL: %s', wml_base_url)
logger.info('WML instance ID: %s', wml_instance_id)

wml_credentials = {
    "apikey": wml_api_key,
    "instance_id": wml_instance_id,
    "url": wml_base_url,
}

iam_token_endpoint = 'https://iam.cloud.ibm.com/identity/token'


def _get_token():
    data = {
        'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
        'apikey': wml_credentials['apikey']
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(iam_token_endpoint, data=data, headers=headers)
    return response.json()['access_token']


def score(token, algorithm, start_date, country, predict_range, s, i, r):
    headers = {'Authorization': 'Bearer ' + token}
    payload = {
        "fields": ["algorithm", "start_date", "country", "predict_range", "S0", "I0", "R0"],
        "values": [[algorithm, start_date.strftime('%-m/%-d/%y'), country, predict_range, s, i, r]]
    }
    logger.info('Scoring with payload: %s', json.dumps(payload))
    response = requests.post(wml_scoring_url, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
    else:
        raise Exception('Scoring error [{}]: {}'.format(response.status_code, response.text))
    n_days = len(result['values'])
    index = [(start_date + timedelta(days=i)).strftime('%d/%m/%y') for i in range(n_days)]
    return pd.DataFrame(result['values'], columns=result['fields'], index=index)


def serve_layout():
    token = _get_token()
    # predict_range = 14
    # sir_result = score(token, 'SIR', datetime(2020, 3, 3), 'Poland', predict_range, 10_000, 20, 10)
    # logistic_result = score(token, 'LOGISTIC', datetime(2020, 3, 3), 'Poland', predict_range, 10_000, 20, 10)
    calibration_result = score(token, 'CALIBRATION', datetime(2020, 1, 22), 'Poland', 40, 10_000, 20, 10)

    # days = list(sir_result.index)
    days = list(calibration_result.index)

    calibration_result['ActualChange'] = calibration_result['Actual'] - calibration_result['Actual'].shift(1, fill_value=0)
    calibration_result['PredictedChange'] = calibration_result['Predicted'] - calibration_result['Predicted'].shift(1, fill_value=0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=days, y=calibration_result['PredictedChange'], name='Predicted Change', opacity=0.5),
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(x=days, y=calibration_result['ActualChange'], name='Actual Change', opacity=0.5),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=days, y=calibration_result['Predicted'], name='Calibration'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=days, y=calibration_result['Actual'], name='Actual', mode="markers", marker=dict(size=8)),
        secondary_y=False,
    )

    fig.update_layout(
        title="Prediction of confirmed cases for Poland",
        template="plotly_dark",
        height=900
    )

    fig.update_xaxes(title_text="Date")

    fig.update_yaxes(title_text="Total confirmed cases", secondary_y=False, range=[0, 6000])
    fig.update_yaxes(title_text="New cases per day", secondary_y=True, range=[0, 1000])

    # fig = go.Figure(
    #     data=[
    #         go.Scatter(x=days, y=sir_result['I'], name='SIR'),
    #         go.Scatter(x=days, y=logistic_result['I'], name='Logistic'),
    #     ],
    #     layout=go.Layout(
    #         title="COVID19 infected prediction in Poland",
    #         template="plotly_dark",
    #         height=600
    #     )
    # )

    return html.Div(children=[
        html.H1(children='COVID-19 Predictions with Watson Machine Learning'),


        dcc.Graph(
            id='example-graph',
            figure=fig
        )
    ])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = serve_layout

if __name__ == '__main__':
    app.run_server(debug=(not is_cf_instance), port=port, host=host)
