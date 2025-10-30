import pandas as pd
import pickle as pkl
from sklearn import pipeline
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def load_and_process_data(path):   
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.replace(' ','_')

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    # there was '_' for this column if there is no data. So we converted it to nan
    df.totalcharges = pd.to_numeric(df.totalcharges,errors = 'coerce')
    df.totalcharges = df.totalcharges.fillna(0)
    
    # will only modify one categorical variable 'churn'
    df.churn = (df.churn == 'yes').astype('int64')
    
    return df

def train(df):
    numerical = ['tenure','monthlycharges','totalcharges']
    categorical = [ 'gender', 'seniorcitizen', 'partner', 'dependents',
            'phoneservice', 'multiplelines', 'internetservice',
        'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
        'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
        'paymentmethod']


     # create pipeline: DictVectorizer transforms list-of-dicts -> feature matrix,
    # LogisticRegression is the final estimator
    pipeline = make_pipeline(
        DictVectorizer(sparse=False),
        LogisticRegression(C=0.8, solver='lbfgs', max_iter=5000)
    )

    dicts = df[categorical + numerical].to_dict(orient='records')
    y = df.churn.values
    pipeline.fit(dicts, y)

    # return the fitted pipeline (you can call pipeline.predict(...) or pipeline.predict_proba(...))
    return pipeline
def save_model(pipeline):
    with open('model.pkl', 'wb') as f_out:
        pkl.dump(pipeline, f_out)
    print("model saved to model.pkl")

if __name__ == '__main__':
    df = load_and_process_data('/workspaces/machine_learning_zoomcamp/05-deployment/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    pipeline = train(df)
    save_model(pipeline)
    print("all done")
    
    print('pandas version:', pd.__version__)
    print('scikit-learn version:', sklearn.__version__)
    print('pipeline steps:', pipeline.named_steps)