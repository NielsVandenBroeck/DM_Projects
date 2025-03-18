import pandas as pd
import numpy as np

def prune_dataset(df, column, param, random=False):
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Filter by InvoiceDate
    if column == "InvoiceDate":
        min_date, max_date = df["InvoiceDate"].min(), df["InvoiceDate"].max()

        if random:
            start_date = np.random.choice(pd.date_range(min_date, max_date - pd.Timedelta(days=param)))
            df = df[(df["InvoiceDate"] >= start_date) & (df["InvoiceDate"] < start_date + pd.Timedelta(days=param))]
        else:
            latest_date = df["InvoiceDate"].max()
            df = df[df["InvoiceDate"] >= latest_date - pd.Timedelta(days=param)]

    # Filter by Customer ID
    elif column == "Customer ID":
        unique_customers = df["Customer ID"].unique()
        if random:
            selected_customers = np.random.choice(unique_customers, min(param, len(unique_customers)), replace=False)
        else:
            selected_customers = df["Customer ID"].value_counts().nlargest(param).index
        df = df[df["Customer ID"].isin(selected_customers)]

    # Filter by InvoiceNo
    elif column == "InvoiceNo":
        unique_invoices = df["InvoiceNo"].unique()
        if random:
            selected_invoices = np.random.choice(unique_invoices, min(param, len(unique_invoices)), replace=False)
        else:
            selected_invoices = df["InvoiceNo"].value_counts().nlargest(param).index
        df = df[df["InvoiceNo"].isin(selected_invoices)]

    return df


def categorize_time(hour):
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 24:
        return "Evening"
    else:
        return "Night"


def fill_in_missing_values(df):
    # Remove negative quantities, assuming these products have been returned.
    df = df.loc[df["Quantity"] >= 0].copy()

    # Fill in missing values with default values
    fill_na = {
        "Description": "No description available",
        "Country": "Unknown",
        "Customer ID": 0,
        "Quantity": 0,
        "InvoiceDate": df["InvoiceDate"].min()
    }
    df.fillna(fill_na, inplace=True)

    # Remove all transactions without InvoiceNo and StockCode since these are important info
    df.dropna(subset=["Invoice"], inplace=True)
    df.dropna(subset=["StockCode"], inplace=True)

    # Check for Price if missing
    def fill_price(row):
        if pd.isna(row["Price"]):
            # Find other prices for the same StockCode
            stock_prices = df[df["StockCode"] == row["StockCode"]]["Price"].dropna()

            if not stock_prices.empty:
                return stock_prices.median()
            else:
                print("sad")
                return np.nan
        return row["Price"]

    df["Price"] = df.apply(fill_price, axis=1)
    return df



df = pd.read_csv("../datasets/retail.csv")

df = fill_in_missing_values(df)

df = prune_dataset(df, "InvoiceDate", 5, False)

# Convert InvoiceDate to datetime format
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df["TimeCategory"] = df["InvoiceDate"].dt.hour.apply(categorize_time)

# Categorize UnitPrice into bins
df["PriceCategory"] = pd.qcut(df["Price"], q=3, labels=["Cheap", "Mid-Range", "Expensive"], duplicates="drop")

# Categorize Quantity into bins
df["QuantityCategory"] = pd.qcut(df["Quantity"], q=3, labels=["Small", "Medium", "Large"], duplicates="drop")


# Save the cleaned dataset
df.to_csv("../datasets/cleaned_retail.csv", index=False)

print("Preprocessing complete. Cleaned dataset saved as 'cleaned_retail.csv'.")
