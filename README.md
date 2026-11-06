# 🧠 Head Brain Analysis Case Study

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Regression](https://img.shields.io/badge/Machine_Learning-Linear_Regression-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

A simple data analysis and regression project built with Python, Pandas, Matplotlib, Seaborn, Scikit-learn and Streamlit. The project studies the relationship between head size and recorded brain weight using a structured dataset and applies Linear Regression to make a basic prediction.

The dashboard focuses on three things: exploring the dataset, studying correlation patterns, and using a regression model to estimate brain weight from a head-size measurement.

> **Note:** This is an educational case study using a small historical dataset. It is not a medical, neurological or intelligence assessment tool. Correlation does not establish causation.

---

## ✨ Features

### 1. Dataset Overview
- Preview the dataset.
- View record and column counts.
- Review descriptive statistics.
- Inspect missing values and category counts.

### 2. Correlation Analysis
- Visualize head size against brain weight.
- Display a simple linear trend line.
- Calculate the Pearson correlation coefficient.
- Explore the direction and strength of the linear association.

### 3. Regression & Prediction
- Train a Linear Regression model using head size as the input feature.
- Predict brain weight as the continuous target.
- Evaluate the model using R², MAE and RMSE.
- Compare actual and predicted values visually.
- Enter a head-size measurement to view an estimated brain-weight output.

---

## 🧠 Dataset

The application uses the **Brain Weight by Head Size** dataset, containing 237 observations and four columns.

| Column | Description |
|---|---|
| `Gender` | Gender category code in the source dataset |
| `Age Range` | Age category code in the source dataset |
| `Head Size(cm^3)` | Head size in cubic centimetres |
| `Brain Weight(grams)` | Recorded brain weight in grams |

The regression model uses `Head Size(cm^3)` as its input and `Brain Weight(grams)` as its target. Gender and Age Range are retained for dataset exploration only.

### Download the dataset

1. Open the [headbrain.csv dataset on GitHub](https://github.com/apekshar5/Machine-Learning/blob/master/predict%20the%20brain%20weight/headbrain.csv).
2. Click **Raw**.
3. Save the file as `headbrain.csv`.
4. Place it in the same directory as `app.py`.

You can also run the included downloader:

```bash
python download_dataset.py
```

---

## ⚙️ Project Workflow

1. Load and validate the CSV dataset.
2. Review the dataset and descriptive statistics.
3. Visualize the relationship between head size and brain weight.
4. Calculate the Pearson correlation coefficient.
5. Split the data into training and testing portions (80/20).
6. Train a Linear Regression model using the training portion.
7. Evaluate predictions on the held-out test data.
8. Allow the user to enter a head-size value and view the model estimate.

### Regression metrics

- **R² Score:** Indicates how much variation in the target is explained by the model.
- **MAE:** Average absolute prediction error, measured in grams.
- **RMSE:** Root mean squared prediction error, measured in grams.

No accuracy figure is hard-coded. The metrics are calculated when the app runs.

---

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive dashboard |
| Pandas | Dataset loading and processing |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical plot and regression trend line |
| Scikit-learn | Linear Regression, train-test split and evaluation metrics |

---

## 📁 Project Structure

```text
Head-Brain-Analysis/
│
├── app.py                  # Main Streamlit application
├── headbrain.csv           # Dataset (download separately)
├── download_dataset.py    # Optional dataset downloader
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── LICENSE                 # MIT License
├── .gitignore              # Git exclusions
│
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
│
└── screenshots/            # Dashboard screenshots (optional)
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.10 or newer recommended.
- pip package manager.
- `headbrain.csv` in the project root.

### 1. Clone your repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Head-Brain-Analysis
```

### 2. Create and activate a virtual environment

**Windows Command Prompt / PowerShell:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Git Bash:**

```bash
python -m venv venv
source venv/Scripts/activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Make sure the project directory contains:

```text
headbrain.csv
```

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

---

## ⚠️ Limitations

- The dataset contains only 237 observations.
- The data is historical and may not represent wider populations.
- The model uses a simple one-feature linear relationship.
- A correlation does not prove that one measurement causes another.
- Predictions should not be interpreted as medical or neurological conclusions.

---

## 👨‍💻 Author

**Shivam Ramesh Kurlekar**

GitHub: [@bemo-codes](https://github.com/bemo-codes)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">Built with Python 🐍 and Streamlit 📊</p>
