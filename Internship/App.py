import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your datasets
file_path = 'Indian House Prices.csv'
DF = pd.read_csv(file_path)

# Rename 'No. of Bedrooms' to 'BHK' for consistency
DF.rename(columns={'No. of Bedrooms': 'BHK'}, inplace=True)

# Set a custom color palette for consistent visuals
sns.set_palette("Set2")

# Streamlit layout and title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Housing Price Analysis</h1>", unsafe_allow_html=True)

# Display column names to verify
st.markdown("<h3 style='color: #FF6347;'>Column Names in the Dataset</h3>", unsafe_allow_html=True)
st.write(DF.columns)

# Check the first few rows of the data to verify structure
st.markdown("<h3 style='color: #1E90FF;'>Data Overview</h3>", unsafe_allow_html=True)
st.write(DF.head())

# --- Function to remove outliers based on City and Price ---
def remove_city_outliers(DF):
    data_new = pd.DataFrame()
    for key, subdata in DF.groupby('City'):
        mean = np.mean(subdata['Price'])  # Use 'Price' instead of 'Price_per_sqft'
        std = np.std(subdata['Price'])    # Use 'Price' instead of 'Price_per_sqft'
        new = subdata[(subdata['Price'] > (mean - std)) & (subdata['Price'] <= (mean + std))]
        data_new = pd.concat([data_new, new], ignore_index=True)
    return data_new

# Apply the outlier removal function
Df = remove_city_outliers(DF)
st.markdown("<h3 style='color: #4CAF50;'>Data After Removing City-Based Outliers</h3>", unsafe_allow_html=True)
st.write(Df.head())
st.write(f"Shape of data after outlier removal: {Df.shape}")

# Select cities for violin plot visualization
cities = ['Mumbai', 'Delhi', 'Chennai', 'Banglore', 'Hyderabad', 'Kolkata']

st.markdown("<h3 style='color: #FF6347;'>Price Distribution by City (After Outlier Removal)</h3>", unsafe_allow_html=True)

# Plot violin plots for selected cities after outlier removal
for city in cities:
    df1 = Df[Df['City'] == city]
    plt.figure(figsize=(10, 8))
    sns.violinplot(df1['Price'], color='orange')  # Use 'Price' instead of 'Price_per_sqft'
    plt.title(f'Data distribution of Price for {city} after Outlier Removal', fontsize=16, color='#FF6347')
    st.pyplot(plt)

# Plot 1: Prices distribution based on city and number of bedrooms
st.markdown("<h3 style='color: #FF6347;'>Price Distribution by City and BHK</h3>", unsafe_allow_html=True)
if 'BHK' in DF.columns and 'City' in DF.columns:
    plt.figure(figsize=(12, 8))
    sns.histplot(data=DF, x='City', hue='BHK', multiple='dodge', shrink=.9)
    plt.title('Prices Distribution by City and BHK', fontsize=16, color='#4CAF50')
    st.pyplot(plt)
else:
    st.error("The column 'BHK' or 'City' was not found in the dataset.")

# Plot 2: BHK (Bedroom, Hall, Kitchen) count plot
st.markdown("<h3 style='color: #1E90FF;'>Count of Bedrooms (BHK) Distribution</h3>", unsafe_allow_html=True)
plt.figure(figsize=(12, 8))
sns.countplot(x='BHK', data=DF, palette="Set3")
plt.xlabel("No. of Bedrooms (BHK)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("BHK (Bedroom, Hall, Kitchen) Distribution", fontsize=16, color='#FF6347')

# Add labels to each bar
for p in plt.gca().patches:
    plt.gca().annotate('{:.0f}'.format(p.get_height()), (p.get_x() + 0.3, p.get_height() + 0.01))

# Show the plot in Streamlit
st.pyplot(plt)

# Plot 3: City-wise count plot
st.markdown("<h3 style='color: #FF6347;'>City-wise Property Count Distribution</h3>", unsafe_allow_html=True)
plt.figure(figsize=(12, 8))
sns.countplot(x='City', data=DF, palette="Set2")

# Add labels and title
plt.xlabel("City", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("City-wise Distribution of Properties", fontsize=16, color='#4CAF50')

# Add labels to each bar
for p in plt.gca().patches:
    plt.gca().annotate('{:.0f}'.format(p.get_height()), (p.get_x() + 0.2, p.get_height() + 0.5), color='black')

# Show the final plot in Streamlit
st.pyplot(plt)

# --- Model Evaluation Results ---
st.markdown("<h3 style='color: #FF6347;'>Model Evaluation Metrics</h3>", unsafe_allow_html=True)
metrics_text = """\
MAE         MSE         RMSE       R-squared
Linear Regression         9.286808  203.810769  14.276231   0.947419
Decision Tree Regressor   1.829392   55.007797   7.416724   0.985808
Random Forest Regression  1.536301   43.897245   6.625500   0.988675
"""
best_model_text = "----->> BEST MODEL IS -----> > > > {'Random Forest Regression'}"

# Display the metrics
st.text(metrics_text)
st.text(best_model_text)

# Footer text with some styling
st.markdown("<h4 style='text-align: center; color: #808080;'>Housing Data Analysis</h4>", unsafe_allow_html=True)
