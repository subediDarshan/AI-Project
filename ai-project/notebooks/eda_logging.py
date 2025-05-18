import mlflow
import dagshub
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

CONFIG = {
    "data_path": "data/external/train.csv",
    "mlflow_tracking_uri": "",
    "dagshub_repo_owner": "",
    "dagshub_repo_name": "",
    "experiment_name": ""
}

mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

train = pd.read_csv("data/raw/merged_train.csv")
test = pd.read_csv("data/raw/merged_test.csv")



with mlflow.start_run():
    # 1. Correlation heatmap - train
    train1 = train.select_dtypes(include=['float64', 'float32','int32','int64'])
    test1 = test.select_dtypes(include=['float64', 'float32','int32','int64'])
    plt.figure(figsize=(20, 10))
    sns.set(font_scale=1.1)
    sns.heatmap(train1.corr(), linewidths=0.5, annot=True, cmap='coolwarm')
    mlflow.log_figure(plt.gcf(), "plots/train_corr_heatmap.png")
    plt.close()

    # 2. Correlation heatmap - test
    plt.figure(figsize=(13, 6))
    sns.set(font_scale=1.1)
    sns.heatmap(test1.corr().round(2), linewidths=0.25, annot=True, cmap='coolwarm')
    mlflow.log_figure(plt.gcf(), "plots/test_corr_heatmap.png")
    plt.close()

    # 3. Store distribution (bar chart)

    plt.figure(figsize=(14, 6))
    sns.set(palette="colorblind", font_scale=1.5)
    train['Store'].value_counts(normalize=True).plot(kind='bar')
    plt.title("Normalized Store Distribution")
    mlflow.log_figure(plt.gcf(), "plots/store_distribution.png")
    plt.close()

    # 4. Weekly Sales Distribution
    plt.figure(dpi=65)
    sns.displot(train["Weekly_Sales"])
    plt.title("Weekly Sales Distribution")
    mlflow.log_figure(plt.gcf(), "plots/weekly_sales_distplot.png")
    plt.close()

    # 5. Total Weekly Sales by Store
    fig = train.groupby('Store').agg({'Weekly_Sales': "sum"}).reset_index().sort_values('Weekly_Sales', ascending=False).plot(kind='bar')
    plt.title("Total Weekly Sales per Store")
    mlflow.log_figure(fig.get_figure(), "plots/weekly_sales_by_store.png")
    plt.close()

    # 6. Weekly Sales by Type (barplot)
    plt.figure(figsize=(7, 5), dpi=65)
    sns.set_style('ticks')
    sns.barplot(y=train["Weekly_Sales"], x=train["Type"], palette='colorblind')
    plt.title("Weekly Sales by Store Type")
    sns.despine()
    mlflow.log_figure(plt.gcf(), "plots/weekly_sales_by_type.png")
    plt.close()

    # 7. Weekly Sales by Dept (pointplot)
    plt.figure(figsize=(18, 7))
    sns.set_style('darkgrid')
    sns.pointplot(x='Dept', y='Weekly_Sales', data=train)
    plt.grid()
    plt.title("Weekly Sales by Department")
    mlflow.log_figure(plt.gcf(), "plots/weekly_sales_by_dept.png")
    plt.close()

print("✅ All plots successfully logged to MLflow (Dagshub).")