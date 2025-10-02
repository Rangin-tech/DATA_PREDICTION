🤖 Universal Data Analyzer & Predictor

A universal data analyzer and predictive modeling app built with Python and Streamlit. Upload any CSV or Excel file, explore your data, visualize distributions, detect outliers, and run predictions using regression or classification models — all in a web interface.


🚀 Features

Upload CSV or Excel files for analysis

Automatic data cleaning and preprocessing

Numeric & categorical data exploration

Histograms, scatter plots, and correlation matrices

Missing value and outlier detection

Predictive modeling:

Regression (Linear Regression)

Classification (Random Forest)

Custom input for predictions

Download processed dataset as CSV


📂 Project Structure

Universal_Data_Analyzer/

│── app.py              # Main Streamlit app
│── requirements.txt    # Project dependencies
│── README.md           # Project documentation


💻 Installation

Clone the repository:

git clone https://github.com/Rangin-tech/DATA_PREDICTION/blob/main/README.md
cd Universal_Data_Analyzer


Install dependencies:

pip install -r requirements.txt

▶️ Run the App
streamlit run app.py


Open your browser at http://localhost:8501 to use the app.

📝 Usage

Upload your CSV or Excel file.

Explore the dataset using stats, histograms, scatter plots, and correlation.

Detect missing values and outliers.

Select a target column to run regression or classification.

Input custom values to make predictions.

Download the processed dataset.


⚙️ Dependencies

Python 3.8+

Streamlit

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Install all with:

pip install -r requirements.txt
