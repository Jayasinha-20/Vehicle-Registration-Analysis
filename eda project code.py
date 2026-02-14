import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("C:/Users/HP/Downloads/Motor_Vehicle_Registrations_Dashboard_data (1).csv")

# 1. DATA CLEANING
print("Initial Dataset Shape:", df.shape)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display basic info
print("\nData Info:")
print(df.info())

# Drop rows with any missing numeric vehicle data
vehicle_cols = ['Auto', 'Bus', 'Truck', 'Motorcycle']
df_clean = df.dropna(subset=vehicle_cols).copy()  # .copy() avoids SettingWithCopyWarning

# Add a column for total vehicles
df_clean.loc[:, 'Total Vehicles'] = df_clean[vehicle_cols].sum(axis=1)

print("\nCleaned Dataset Shape:", df_clean.shape)

# 2. BASIC STATISTICS
print("\nSummary Statistics:")
print(df_clean.describe(include='all'))

# 3. TOP 20 VEHICLE REGISTRATION YEARS
yearly = df_clean.groupby('year')['Total Vehicles'].sum().reset_index()
top10_years = yearly.sort_values(by='Total Vehicles', ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(data=top10_years, x='year', y='Total Vehicles', hue='year', palette='viridis', legend=False, errorbar=None)
plt.title('Top 10 Years by Total Vehicle Registrations')
plt.ylabel('Total Vehicles')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. TOP STATES BY TOTAL VEHICLES
top_states = df_clean.groupby('state')['Total Vehicles'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(y=top_states.index, x=top_states.values, hue=top_states.index, palette='magma', legend=False)
plt.title("Top 10 States by Total Vehicle Registrations")
plt.xlabel("Total Vehicles")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# 5. VEHICLE TYPE TRENDS OVER YEARS
yearly_vehicles = df_clean.groupby('year')[vehicle_cols].sum().reset_index()
plt.figure(figsize=(12, 6))
for col in vehicle_cols:
    plt.plot(yearly_vehicles['year'], yearly_vehicles[col], marker='o', label=col)

plt.title('Trend of Vehicle Types Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Vehicles')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 6. VEHICLE TYPE SHARE (OVERALL)
vehicle_totals = df_clean[vehicle_cols].sum().sort_values(ascending=False)
plt.figure(figsize=(8, 6))
plt.pie(vehicle_totals, labels=vehicle_totals.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2'))
plt.title("Share of Each Vehicle Type (Overall)")
plt.tight_layout()
plt.show()

# 7. TOP 10 STATES BY MOTORCYCLE REGISTRATIONS
top_motorcycle_states = df_clean.groupby('state')['Motorcycle'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_motorcycle_states.values,
            y=top_motorcycle_states.index,
            hue=top_motorcycle_states.index,
            palette='cubehelix',
            legend=False)
plt.title("Top 10 States by Motorcycle Registrations")
plt.xlabel("Number of Motorcycles")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# 8. INSIGHTS
print("\n--- INSIGHT SUMMARY ---")
print(f"Years covered: {df_clean['year'].nunique()}")
print(f"States monitored: {df_clean['state'].nunique()}")
print(f"Total vehicle registrations: {df_clean['Total Vehicles'].sum():,.0f}")
most_common_type = df_clean[vehicle_cols].sum().idxmax()
print(f"Most registered vehicle type: {most_common_type}")



# 9. BOX PLOT OF VEHICLE TYPES BY STATE (Top 5 States for Simplicity)
top5_states = df_clean['state'].value_counts().head(5).index
df_top_states = df_clean[df_clean['state'].isin(top5_states)]

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_top_states, x='state', y='Motorcycle', hue='state', palette='coolwarm', legend=False)

plt.title("Distribution of Motorcycle Registrations in Top 5 States")
plt.xlabel("State")
plt.ylabel("Number of Motorcycles")
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. HEATMAP OF VEHICLE TYPE CORRELATIONS
plt.figure(figsize=(8, 6))
corr_matrix = df_clean[vehicle_cols + ['Total Vehicles']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title("Correlation Heatmap of Vehicle Types")
plt.tight_layout()
plt.show()







