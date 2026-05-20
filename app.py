import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# LOAD DATASET & MODEL
# -----------------------------
df = pd.read_csv("./CO2 Emissions_Canada.csv")
print(df.columns)
target_column = "CO2 Emissions(g/km)"
model = joblib.load("model.pkl")

st.title("🚗 Vehicle Emission Prediction App")

# -----------------------------
# A. DATASET OVERVIEW
# -----------------------------
st.header("A. Dataset Overview")

if st.checkbox("Show Dataset"):
    st.write(df)

if st.checkbox("Show Summary Statistics"):
    st.write(df.describe())

if st.checkbox("Show Missing Values"):
    st.write(df.isnull().sum())

# -----------------------------
# B. VISUAL ANALYTICS
# -----------------------------
st.header("B. Visual Analytics")

# Univariate Analysis
st.subheader("Univariate Analysis")

uni_col = st.selectbox("Select Column for Histogram", df.columns)

fig1, ax1 = plt.subplots()

# Handle numeric and categorical columns
if df[uni_col].dtype == "object":
    df[uni_col].value_counts().plot(kind="bar", ax=ax1)
else:
    sns.histplot(df[uni_col], kde=True, ax=ax1)

st.pyplot(fig1)

# Bivariate Analysis
st.subheader("Bivariate Analysis")

x_col = st.selectbox("Select X-axis", df.columns, key="x")
y_col = st.selectbox("Select Y-axis", df.columns, key="y")

fig2, ax2 = plt.subplots()

# Only scatterplot for numeric columns
if df[x_col].dtype != "object" and df[y_col].dtype != "object":
    sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax2)
    st.pyplot(fig2)
else:
    st.warning("Please select numeric columns for scatter plot.")

# Correlation Heatmap
st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=['int64', 'float64'])

fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax3)

st.pyplot(fig3)

# -----------------------------
# C. PREDICTION MODULE
# -----------------------------
# -----------------------------
# C. PREDICTION MODULE
# -----------------------------
st.header("C. Prediction Module")

st.write("Enter Vehicle Details")

# Correct target column name
target_column = "CO2 Emissions(g/km)"   # CHANGE if needed

input_data = {}

# Take inputs according to datatype
for column in df.columns:

    # Skip target column
    if column == target_column:
        continue

    # TEXT / CATEGORICAL COLUMNS
    if df[column].dtype == "object":

        input_data[column] = st.selectbox(
            f"Select {column}",
            df[column].unique()
        )

    # NUMERIC COLUMNS
    else:

       try:
         default_value = float(
            pd.to_numeric(df[column], errors='coerce').mean()
        )

       except:
        default_value = 0.0

    input_data[column] = st.number_input(
        f"Enter {column}",
        value=default_value
    )

# Predict Button
if st.button("Predict Emission"):

    # Convert input to dataframe
    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    input_encoded = pd.get_dummies(input_df)

    # Encode training features
    train_features = pd.get_dummies(
        df.drop(target_column, axis=1)
    )

    # Match columns
    input_encoded = input_encoded.reindex(
        columns=train_features.columns,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_encoded)

    st.success(
        f"Predicted CO2 Emission: {prediction[0]:.2f}"
    )

    # -----------------------------
# D. GRAPHICAL OUTPUT
# -----------------------------

st.header("D. Graphical Output")

if st.button("Show Prediction Comparison"):

    sample = df.sample(10)

    fig4, ax4 = plt.subplots(figsize=(12,5))

    # Vehicle names for X-axis
    if "Make" in df.columns and "Model" in df.columns:

        vehicle_names = sample["Make"] + " " + sample["Model"]

    elif "Make" in df.columns:

        vehicle_names = sample["Make"]

    else:

        vehicle_names = range(1, 11)

    # Plot graph
    ax4.plot(
        vehicle_names,
        sample[target_column].values,
        marker='o'
    )

    ax4.set_xlabel("Vehicle Name")
    ax4.set_ylabel("CO2 Emission")
    ax4.set_title("Vehicle CO2 Emission Comparison")

    # Rotate names for better visibility
    plt.xticks(rotation=90)

    st.pyplot(fig4)