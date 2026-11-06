#############################################################################
#
#   Importing required libraries
#
#############################################################################
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#############################################################################
#
#   Project constants
#
#############################################################################
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "headbrain.csv"
RANDOM_STATE = 42
HEAD_SIZE_COLUMN = "Head Size(cm^3)"
BRAIN_WEIGHT_COLUMN = "Brain Weight(grams)"
REQUIRED_COLUMNS = ["Gender", "Age Range", HEAD_SIZE_COLUMN, BRAIN_WEIGHT_COLUMN]

#############################################################################
#
#   Function name:      inject_custom_css
#   Input:              None
#   Description:        Applies simple styling to the Streamlit dashboard.
#   Output:             None
#   Date:               07/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def inject_custom_css():
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(135deg, #07111f 0%, #0b1b2b 58%, #102b38 100%);
                color: #edf6f7;
            }
            [data-testid="stSidebar"] {
                background-color: #091725;
                border-right: 1px solid rgba(45, 212, 191, 0.18);
            }
            .hero-title {
                color: #e8ffff;
                font-size: 2.6rem;
                font-weight: 800;
                text-align: center;
                margin-bottom: 0.2rem;
            }
            .hero-subtitle {
                color: #91b8c1;
                text-align: center;
                font-size: 1rem;
                margin-bottom: 1.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


#############################################################################
#
#   Function name:      load_data
#   Input:              data_path (string)
#   Description:        Loads the local CSV file and validates required columns.
#   Output:             Validated Pandas DataFrame
#   Date:               07/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
@st.cache_data(show_spinner=False)
def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.columns = [str(column).strip() for column in df.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing these columns: " + ", ".join(missing_columns)
        )

    df = df[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[HEAD_SIZE_COLUMN, BRAIN_WEIGHT_COLUMN])
    if df.empty:
        raise ValueError("No usable records were found in the dataset.")
    return df


#############################################################################
#
#   Function name:      train_and_evaluate_model
#   Input:              df (DataFrame)
#   Description:        Trains a simple Linear Regression model using head size
#                       to estimate brain weight and evaluates it on test data.
#   Output:             Trained model, test metrics, test values and predictions
#   Date:               07/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def train_and_evaluate_model(df: pd.DataFrame):
    X = df[[HEAD_SIZE_COLUMN]]
    y = df[BRAIN_WEIGHT_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "R² Score": r2_score(y_test, predictions),
        "MAE (grams)": mean_absolute_error(y_test, predictions),
        "RMSE (grams)": np.sqrt(mean_squared_error(y_test, predictions)),
    }
    return model, metrics, X_test, y_test, predictions


#############################################################################
#
#   Function name:      plot_relationship
#   Input:              df (DataFrame)
#   Description:        Displays the relationship between head size and brain
#                       weight with a simple linear trend line.
#   Output:             Matplotlib figure
#   Date:               07/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def plot_relationship(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(
        data=df,
        x=HEAD_SIZE_COLUMN,
        y=BRAIN_WEIGHT_COLUMN,
        scatter_kws={"alpha": 0.75, "s": 55},
        line_kws={"color": "#f59e0b", "linewidth": 2},
        ax=ax,
    )
    ax.set_title("Head Size vs Brain Weight")
    ax.set_xlabel("Head Size (cm³)")
    ax.set_ylabel("Brain Weight (grams)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


#############################################################################
#
#   Function name:      render_dataset_tab
#   Input:              df (DataFrame)
#   Description:        Displays dataset preview, summary statistics and counts.
#   Output:             None
#   Date:               07/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def render_dataset_tab(df: pd.DataFrame):
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Records", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing values", int(df.isna().sum().sum()))

    st.markdown("#### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.describe().T, use_container_width=True)
    with right:
        st.markdown("#### Dataset Categories")
        st.write("Gender values (as coded in the source dataset)")
        st.dataframe(df["Gender"].value_counts().sort_index())
        st.write("Age Range values (as coded in the source dataset)")
        st.dataframe(df["Age Range"].value_counts().sort_index())


#############################################################################
#
#   Function name:      render_correlation_tab
#   Input:              df (DataFrame)
#   Description:        Shows the head-size and brain-weight relationship and
#                       their Pearson correlation coefficient.
#   Output:             None
#   Date:               07/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def render_correlation_tab(df: pd.DataFrame):
    st.subheader("Correlation Analysis")
    correlation = df[HEAD_SIZE_COLUMN].corr(df[BRAIN_WEIGHT_COLUMN])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Head Size / Brain Weight Correlation", f"{correlation:.3f}")
        st.caption(
            "Pearson correlation ranges from -1 to +1. It describes the strength "
            "and direction of a linear association, not causation."
        )
    with col2:
        st.pyplot(plot_relationship(df), use_container_width=True)


#############################################################################
#
#   Function name:      render_regression_tab
#   Input:              model, metrics, X_test, y_test, predictions
#   Description:        Displays Linear Regression evaluation metrics and
#                       compares actual brain weights with model predictions.
#   Output:             None
#   Date:               07/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def render_regression_tab(model, metrics, y_test, predictions):
    st.subheader("Linear Regression Results")
    st.caption("The model uses head size as its input and brain weight as its target.")

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{metrics['R² Score']:.3f}")
    col2.metric("Mean Absolute Error", f"{metrics['MAE (grams)']:.1f} g")
    col3.metric("Root Mean Squared Error", f"{metrics['RMSE (grams)']:.1f} g")

    results = pd.DataFrame(
        {
            "Actual Brain Weight (g)": np.asarray(y_test),
            "Predicted Brain Weight (g)": predictions,
        }
    )
    st.markdown("#### Actual vs Predicted Values")
    st.dataframe(results.head(10).round(1), use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(y_test, predictions, alpha=0.75, color="#0d9488")
    lower = min(float(np.min(y_test)), float(np.min(predictions)))
    upper = max(float(np.max(y_test)), float(np.max(predictions)))
    ax.plot([lower, upper], [lower, upper], "--", color="#f59e0b")
    ax.set_xlabel("Actual Brain Weight (grams)")
    ax.set_ylabel("Predicted Brain Weight (grams)")
    ax.set_title("Actual vs Predicted Brain Weight")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.caption(
        f"Regression equation: Brain Weight = "
        f"{model.coef_[0]:.3f} × Head Size + {model.intercept_:.3f}"
    )


#############################################################################
#
#   Function name:      render_prediction_tab
#   Input:              df (DataFrame), model (LinearRegression)
#   Description:        Accepts a head-size measurement and predicts brain weight.
#   Output:             None
#   Date:               07/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def render_prediction_tab(df: pd.DataFrame, model: LinearRegression):
    st.subheader("Brain Weight Prediction")
    st.info(
        "Enter a head-size value to see the output of the fitted regression model. "
        "This is an educational exercise using a small historical dataset."
    )

    minimum = float(df[HEAD_SIZE_COLUMN].min())
    maximum = float(df[HEAD_SIZE_COLUMN].max())
    default = float(df[HEAD_SIZE_COLUMN].median())

    head_size = st.number_input(
        "Head Size (cm³)",
        min_value=minimum,
        max_value=maximum,
        value=default,
        step=1.0,
    )

    if st.button("Predict Brain Weight", type="primary"):
        input_data = pd.DataFrame({HEAD_SIZE_COLUMN: [head_size]})
        prediction = float(model.predict(input_data)[0])
        st.success(f"Estimated Brain Weight: **{prediction:.1f} grams**")
        st.caption(
            "This is a model estimate from the dataset, not a medical or neurological assessment."
        )


#############################################################################
#
#   Function name:      main
#   Input:              None
#   Description:        Loads the dataset, trains Linear Regression and displays
#                       the three main sections of the analysis dashboard.
#   Output:             None
#   Date:               07/11/2026
#   Author:             Shivam Ramesh Kurlekar
#
#############################################################################
def main():
    st.set_page_config(
        page_title="Head & Brain Analysis",
        page_icon="🧠",
        layout="wide",
    )
    inject_custom_css()

    st.markdown(
        "<div class='hero-title'>🧠 Head & Brain Analysis</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='hero-subtitle'>Regression and correlation analysis on a structured dataset</div>",
        unsafe_allow_html=True,
    )

    if not DATA_PATH.exists():
        st.error("Dataset not found. Place `headbrain.csv` beside `app.py`.")
        st.code("Head-Brain-Analysis/\n├── app.py\n└── headbrain.csv")
        st.stop()

    try:
        df = load_data(str(DATA_PATH))
    except Exception as error:
        st.error(f"Unable to load the dataset: {error}")
        st.stop()

    with st.spinner("Training the Linear Regression model..."):
        model, metrics, X_test, y_test, predictions = train_and_evaluate_model(df)

    st.sidebar.header("Project Information")
    st.sidebar.write(f"Dataset: `{DATA_PATH.name}`")
    st.sidebar.write(f"Records: **{len(df)}**")
    st.sidebar.write("Model: **Linear Regression**")

    tab1, tab2, tab3 = st.tabs(
        ["📊 Dataset Overview", "🔎 Correlation Analysis", "📈 Regression & Prediction"]
    )

    with tab1:
        render_dataset_tab(df)
    with tab2:
        render_correlation_tab(df)
    with tab3:
        render_regression_tab(model, metrics, y_test, predictions)
        st.divider()
        render_prediction_tab(df, model)

    st.divider()
    st.caption(
        "Educational case study only. Correlation does not establish causation. "
        "The dataset is small and historical; predictions should not be used for medical purposes."
    )


#############################################################################
#
#   Starter of main
#
#############################################################################
if __name__ == "__main__":
    main()
