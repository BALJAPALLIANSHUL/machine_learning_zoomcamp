
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_PATH = "data/raw/data.csv"
MODEL_PATH = "models/model.bin"

def main():
    df = pd.read_csv(DATA_PATH)

    y = df['target']
    X = df.drop(columns=['target'])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    dv = DictVectorizer(sparse=False)
    X_train_dict = X_train.to_dict(orient='records')
    X_val_dict = X_val.to_dict(orient='records')

    X_train_vec = dv.fit_transform(X_train_dict)
    X_val_vec = dv.transform(X_val_dict)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_val_vec)
    acc = accuracy_score(y_val, y_pred)

    print(f"Validation accuracy: {acc:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((dv, model), f)

if __name__ == "__main__":
    main()
