
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

def extract_training_data(xls_path):
    try:
        xls = pd.ExcelFile(xls_path)
        if "Template" not in xls.sheet_names:
            return pd.DataFrame()
        df = xls.parse("Template")
        df.columns = df.iloc[0]
        df = df[1:]
        df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()
        cols = [
            "Vendor Product Category (Required)",
            "Product Title (Required)",
            "Product Group (Required)",
        ]
        for c in cols:
            if c not in df.columns:
                return pd.DataFrame()
        df = df[cols].dropna()
        df["text"] = df["Vendor Product Category (Required)"] + " " + df["Product Title (Required)"]
        return df
    except Exception as e:
        print(f"Failed to process {xls_path}: {e}")
        return pd.DataFrame()

def main(input_folder="training_data", output_model="pg_classifier_latest.joblib"):
    all_dfs = []
    for file in Path(input_folder).glob("*.xls*"):
        df = extract_training_data(file)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("No valid training data found.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Filter PGs with at least 5 samples
    pg_counts = full_df["Product Group (Required)"].value_counts()
    valid_pgs = pg_counts[pg_counts >= 5].index
    filtered_df = full_df[full_df["Product Group (Required)"].isin(valid_pgs)]

    X = filtered_df["text"]
    y = filtered_df["Product Group (Required)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, output_model)

    print(f"Model trained and saved to: {output_model}")
    print("Validation Report:")
    print(classification_report(y_test, model.predict(X_test)))

if __name__ == "__main__":
    main()
