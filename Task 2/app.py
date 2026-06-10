import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
st.set_page_config(
    page_title="Netflix Dashboard",
    page_icon="🎬",
    layout="wide"
)
st.title("🎬 Netflix Data Analysis Dashboard")
st.markdown("Interactive dashboard using Streamlit")
df = pd.read_csv("archive/netflix_titles.csv")
df.drop_duplicates(inplace=True)
df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Not Available')
df['country'] = df['country'].fillna(df['country'].mode()[0])
df['rating'] = df['rating'].fillna(df['rating'].mode()[0])
# Convert date column
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
# Create extra columns
df['added_year'] = df['date_added'].dt.year
df['added_month'] = df['date_added'].dt.month_name()
st.sidebar.header("🔍 Filters")
type_filter = st.sidebar.multiselect(
    "Select Type",
    options=df['type'].unique(),
    default=df['type'].unique()
)
country_filter = st.sidebar.multiselect(
    "Select Country",
    options=df['country'].unique(),
    default=df['country'].unique()[:10]
)
# Filter Data
filtered_df = df[
    (df['type'].isin(type_filter)) &
    (df['country'].isin(country_filter))
]
total_titles = len(filtered_df)
movies = len(filtered_df[filtered_df['type'] == 'Movie'])
tvshows = len(filtered_df[filtered_df['type'] == 'TV Show'])
countries = filtered_df['country'].nunique()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Titles", total_titles)
col2.metric("Movies", movies)
col3.metric("TV Shows", tvshows)
col4.metric("Countries", countries)
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Movies vs TV Shows")
    fig1, ax1 = plt.subplots()
    sns.countplot(
        x='type',
        data=filtered_df,
        ax=ax1
    )
    st.pyplot(fig1)
with col2:
    st.subheader("Ratings Distribution")
    rating_counts = filtered_df['rating'].value_counts().head(10)
    fig2, ax2 = plt.subplots()
    ax2.pie(
        rating_counts.values,
        labels=rating_counts.index,
        autopct='%1.1f%%'
    )
    st.pyplot(fig2)
col3, col4 = st.columns(2)
with col3:
    st.subheader("Content Added Over Years")
    yearly_data = filtered_df['added_year'].value_counts().sort_index()
    fig3, ax3 = plt.subplots()
    ax3.plot(
        yearly_data.index,
        yearly_data.values
    )
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Content Count")
    st.pyplot(fig3)
with col4:
    st.subheader("Release Year Distribution")
    fig4, ax4 = plt.subplots()
    ax4.hist(
        filtered_df['release_year'],
        bins=20
    )
    ax4.set_xlabel("Release Year")
    ax4.set_ylabel("Frequency")
    st.pyplot(fig4)
st.subheader("Top 10 Countries")
top_countries = filtered_df['country'].value_counts().head(10)
fig5, ax5 = plt.subplots(figsize=(10, 5))
sns.barplot(
    x=top_countries.values,
    y=top_countries.index,
    ax=ax5
)
st.pyplot(fig5)
st.subheader("Correlation Heatmap")
numeric_df = filtered_df.select_dtypes(include=np.number)
fig6, ax6 = plt.subplots(figsize=(8, 5))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm',
    ax=ax6
)
st.pyplot(fig6)
st.subheader("Dataset Preview")
st.dataframe(filtered_df)
st.markdown("---")
st.markdown("Created using Python, Pandas, Seaborn & Streamlit")