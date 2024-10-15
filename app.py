import streamlit as st
import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_iris, load_wine
import matplotlib.pyplot as plt
import time
import psutil

st.set_page_config(
    page_title="Abhay's Machine Learning Comparison App",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'A personal project by abhaypai@vt.edu',
    }
)


def get_memory_usage():
    return psutil.Process().memory_info().rss / (1024 * 1024)  # in MB

def load_demo_dataset(dataset_name):
    if dataset_name == "Diabetes":
        data = load_boston()
        X, y = data.data, data.target
        task = "regression"
    elif dataset_name == "Iris":
        data = load_iris()
        X, y = data.data, data.target
        task = "classification"
    elif dataset_name == "Wine":
        data = load_wine()
        X, y = data.data, data.target
        task = "classification"
    return X, y, task

def run_lazypredict(X, y, task):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    if task == "regression":
        clf = LazyRegressor(verbose=1, ignore_warnings=True, custom_metric=None)
    else:
        clf = LazyClassifier(verbose=1, ignore_warnings=True, custom_metric=None)
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    time_taken = end_time - start_time
    memory_used = end_memory - start_memory
    
    models['Time (s)'] = time_taken / len(models)
    models['Memory (MB)'] = memory_used / len(models)
    progress_bar.progress(1.0)

    return models

def display_results(results):
    
    col1, col2 = st.columns([1,4])
    with col2:
        st.subheader("Model Performance Comparison")
        st.write(results)
    
    col1, col2 = st.columns([0.5,6])
    with col2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        results[['Time (s)']].plot(kind='bar', ax=ax1)
        ax1.set_title("Time Complexity")
        ax1.set_ylabel("Time (s)")
        ax1.tick_params(axis='x', rotation=90)
        
        results[['Memory (MB)']].plot(kind='bar', ax=ax2)
        ax2.set_title("Memory Complexity")
        ax2.set_ylabel("Memory (MB)")
        ax2.tick_params(axis='x', rotation=90)
        
        plt.tight_layout()
        st.pyplot(fig)

st.title("ML Model Complexity Comparison")

tab1, tab2 = st.tabs(["Upload Dataset", "Demo Datasets"])

with tab1:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())
        
        target_column = st.selectbox("Select the target column", data.columns)
        task = st.radio("Select the task type", ["regression", "classification"])
        
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        try:
            if st.button("Run Analysis", key="upload"):
                progress_bar = st.progress(0)
                results = run_lazypredict(X, y, task)
                display_results(results)
        except:
            st.error("This dataset is not suitable for "+task)

with tab2:
    dataset_name = st.selectbox("Choose a demo dataset", ["Diabetes", "Iris", "Wine"])
    if st.button("Run Analysis", key="demo"):
        progress_bar = st.progress(0)
        X, y, task = load_demo_dataset(dataset_name)
        results = run_lazypredict(X, y, task)
        display_results(results)
