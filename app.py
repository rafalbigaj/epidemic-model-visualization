import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import logging
import os
import pandas as pd
from datetime import datetime
from datetime import timedelta
from urllib import parse
from watson_machine_learning_client import WatsonMachineLearningAPIClient

logger = logging.getLogger(__name__)

external_stylesheets = [dbc.themes.DARKLY]

wml_api_key = os.environ['WML_API_KEY']
wml_scoring_url = os.environ['WML_SCORING_URL']
url = parse.urlparse(wml_scoring_url)
wml_base_url = url._replace(path='').geturl()
wml_instance_id = url.path.split('/')[3]

logger.info('WML URL:', wml_base_url)
logger.info('WML instance ID:', wml_instance_id)

wml_credentials = {
    "apikey": wml_api_key,
    "instance_id": wml_instance_id,
    "url": wml_base_url,
}

wml_client = WatsonMachineLearningAPIClient(wml_credentials)


def score(algorithm, start_date, country, predict_range, s, i, r):
    payload = {
        "fields": ["algorithm", "start_date", "country", "predict_range", "S0", "I0", "R0"],
        "values": [[algorithm, start_date.strftime('%-m/%-d/%y'), country, predict_range, s, i, r]]
    }
    logger.debug(payload)
    result = wml_client.deployments.score(wml_scoring_url, payload)
    n_days = len(result['values'])
    index = [(start_date + timedelta(days=i)).strftime('%d/%m/%y') for i in range(n_days)]
    return pd.DataFrame(result['values'], columns=result['fields'], index=index)


wml_client.deployments.list()


def serve_layout():
    predict_range = 14
    sir_result = score('SIR', datetime(2020, 3, 3), 'Poland', predict_range, 10_000, 20, 10)
    logger.info('SIR scoring result:')
    logger.info(sir_result)
    logistic_result = score('LOGISTIC', datetime(2020, 3, 3), 'Poland', predict_range, 10_000, 20, 10)
    logger.info('LOGISTIC scoring result:')
    logger.info(logistic_result)

    days = list(sir_result.index)

    fig = go.Figure(
        data=[
            go.Scatter(x=days, y=sir_result['I'], name='SIR'),
            go.Scatter(x=days, y=logistic_result['I'], name='Logistic'),
            go.Scatter(x=days, y=sir_result['Actual'], name='Actual', mode="markers", marker=dict(size=8)),
        ],
        layout=go.Layout(
            title="COVID19 infected prediction per country",
            template="plotly_dark"
        )
    )

    return html.Div(children=[
        html.H1(children='COVID19 predictions with Watson Machine Learning'),

        # html.Div(children='''
        #   SIR model visualization
        # '''),

        dcc.Graph(
            id='example-graph',
            figure=fig
        )
    ])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = serve_layout

if __name__ == '__main__':
    app.run_server(debug=True)
