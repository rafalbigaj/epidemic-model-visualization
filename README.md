# COVID-19 Predictions with Watson Machine Learning

This is a sample application showing how to visualize prediction results of a machine learning model 
deployed on IBM Cloud. Please refer to the blog
[Serving COVID-19 epidemic models with Watson Machine Learning](https://medium.com/@Lukasz.Cmielowski/378b6fe9407b)
for details how to prepare and deploy such a model. 

## Prerequisites

You'll need the following:
- [IBM Cloud account](https://cloud.ibm.com/registration)
- [IBM Cloud CLI](https://cloud.ibm.com/docs/cli/reference/ibmcloud?topic=cloud-cli-install-ibmcloud-cli)
- Python and Git

Additionally, in the blog section "Serving models as web service" you will find information required to configure the sample application.
To make scoring request you need:
- **scoring url** (in form: `https://<region>.ml.cloud.ibm.com/v3/wml_instances/<instance-id>/deployments/<deployment-id>/online`)
- Watson Machine Learning service instance **API key** (`apikey` from the WML service credentials)

 
## How to run the app locally

Once you deploy the epidemic model to IBM Cloud, you can simple set the following system environment variables:
- `WML_SCORING_URL` - the Watson Machine Learning scoring URL goes here
- `WML_API_KEY` - the Watson Machine Learning instance API key goes here

Make sure all the required packages are installed in your python environment:

```shell script
pip install -e requirements.txt
```

Then you can start the application locally with the following command:

```shell script
python app.py
```

## How to deploy to IBM Cloud

At first fill in the manifest.yml file with the required settings:

```yaml
applications:
 - name: covid19-infected-prediction
   memory: 256M
   command: python app.py
   env:
     WML_SCORING_URL: <the Watson Machine Learning scoring URL goes here>
     WML_API_KEY: <the Watson Machine Learning instance API key goes here>
``` 

Then log in to your IBM Cloud account, and select an API endpoint:

```shell script
ibmcloud login
```

target a cloud foundry org and space:
```shell script
ibmcloud target --cf
```

and finally from the application directory push your app to IBM Cloud:
```shell script
ibmcloud cf push
```

## Visualization

To build interactive dashboard consuming our WebService we have used [Plotly Dash](https://plotly.com/dash/). 
That allowed us to stay in python eco-system and achieve our goals easily.

The sample visualization from `app.py` shows results based on the calibration model predictions.
The actual and predicted total number of cases is on the primary Y axis,
while new cases per day on the secondary Y axis. 

```python
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
```

For details see: [how to make a graph with multiple axes in python](https://plotly.com/python/multiple-axes/).