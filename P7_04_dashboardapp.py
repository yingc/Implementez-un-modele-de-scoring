# visit http://127.0.0.1:8050/ in your web browser.
import pandas as pd

from explainerdashboard import ClassifierExplainer, ExplainerDashboard

from explainerdashboard import *
from explainerdashboard.datasets import *
from explainerdashboard.custom import *

from joblib import load
from explainerdashboard import *

from flask import Flask


app = dash.Dash()
#app = Flask(__name__)
server= app.server

model = load("LightGBMC.joblib")
app_df= pd.read_csv("app_df.csv", sep=',')
feats = [f for f in app_df.columns if f not in ['TARGET','SK_ID_CURR','index','SK_ID_BUREAU','SK_ID_PREV']]
X= app_df[feats]
y = app_df["TARGET"]


fi_df =pd.read_csv("feature_importance_df.csv", sep=',')
cols = fi_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
best_fi = fi_df.loc[fi_df.feature.isin(cols)]

HomeCredit_columns_description = pd.read_csv('HomeCredit_columns_description.csv', sep=',', encoding = 'iso8859_1')
row_description = HomeCredit_columns_description.loc[HomeCredit_columns_description.Row.isin(best_fi.feature), ["Row", "Description"]]
feature_descriptions= dict(zip(row_description.Row,row_description.Description))

explainer = ClassifierExplainer(model, X, y,  X_background=X,
                                target='TARGET', # the name of the target variable (y)
                                shap='tree', # TreeExplainer
                                model_output='probability', # model_output of shap values, ici for classification
                                precision='float32', # save memory by setting lower precision. Default is 'float64'
                                descriptions=feature_descriptions, # adds a table and hover labels to dashboard
                                #idxs = "SK_ID_CURR", # defaults to X.index
                                index_name = "SK_ID_CURR", # defaults to X.index.name
                                labels=['ACCORD Credit', 'REFUS Credit'], n_jobs=-1)

db = ExplainerDashboard(explainer,
                    importances=True,  # True
                    model_summary=False,
                    contributions=True, # indivituel prediction
                    whatif=True,  # true
                    depth=20, # only show 40 features
                    shap_dependence=True,
                    shap_interaction=True,  # true
                    decision_trees=False,
                    server=app,
                    title="Crédit de Consommation"
                    )



if __name__=='__main__':
  app.run_server(debug=True)
  
  
