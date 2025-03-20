from apyori import apriori
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('../datasets/cleaned_retail.csv')

# Convert Invoice to string
df["Invoice"] = df["Invoice"].astype(str)

# Remove duplicates
df = df.drop_duplicates(subset=["Invoice", "StockCode"])

# # Group by Invoice and aggregate StockCodes into a list
# transactions = df.groupby("Invoice")["StockCode"].apply(list).tolist()
#
# # Add TimeCategory to each transaction as an additional item
# time_categories = df.groupby("Invoice")["TimeCategory"].first().tolist()
#
# # Combine StockCodes with TimeCategory
# transactions = [trans + [time] for trans, time in zip(transactions, time_categories)]

transactions = df.groupby("Invoice").apply(
    lambda x: x["StockCode"].tolist() + [x["DayOfWeek"].iloc[0]]
).tolist()

# Run Apriori algorithm with adjusted thresholds
frequent_itemsets = apriori(
    transactions,
    min_support=0.02,  # Adjust as needed
    min_confidence=0.2,
    min_lift=1.0,
    min_length=2
)

# Convert results to a list
results = list(frequent_itemsets)

# Filter out rules where TimeCategory is the same on both sides
filtered_rules = []
for rule in results:
    items = list(rule.items)

    # Ensure TimeCategory appears at most once in the rule
    time_categories = [item for item in items if item in ["Monday", "Tuesday", "WednesDay", "Thursday", "Friday", "Saturday", "Sunday"]]

    if len(time_categories) <= 1:  # Keep rules if only one time-related element exists
        filtered_rules.append(rule)

# Print filtered rules
if filtered_rules:
    for rule in filtered_rules[:15]:
        items = [x for x in rule.items]
        print(f"Rule: {items} | Support: {rule.support:.4f}")

        for ordered_stat in rule.ordered_statistics:
            base = ', '.join(map(str, ordered_stat.items_base))
            add = ', '.join(map(str, ordered_stat.items_add))
            print(
                f"  {base} -> {add} | Confidence: {ordered_stat.confidence:.4f} | Lift: {ordered_stat.lift:.4f}"
            )
        print("-" * 50)
else:
    print("No rules found — try adjusting the support and confidence values!")
