import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# =========================
# main.py
# =========================

import pandas as pd

from data_preprocessing import(
    preprocess_customers,
    preprocess_orders,
    preprocess_order_items,
    preprocess_products,
    preprocess_categories,
    preprocess_reviews,
    merge_all,
    clean_outliers_with_boxplots,
    save_raw_processed_data
)


def main():
    print("🚀 Starting Data Pipeline...\n")

    # =========================
    # 1. Load Raw Data
    # =========================
    customers = pd.read_csv("D:/FinalYearProject/olist ecommerce dataset/olist_customers_dataset.csv")
    orders = pd.read_csv("D:/FinalYearProject/olist ecommerce dataset/olist_orders_dataset.csv")
    order_items = pd.read_csv("D:/FinalYearProject/olist ecommerce dataset/olist_order_items_dataset.csv")
    products = pd.read_csv("D:/FinalYearProject/olist ecommerce dataset/olist_products_dataset.csv")
    categories = pd.read_csv("D:/FinalYearProject/olist ecommerce dataset/product_category_name_translation.csv")
    reviews = pd.read_csv("D:/FinalYearProject/olist ecommerce dataset/olist_order_reviews_dataset.csv")
    print("✅ Data Loaded Successfully!\n")

    # =========================
    # 2. Preprocessing
    # =========================
    customers = preprocess_customers(customers)
    orders = preprocess_orders(orders)
    order_items = preprocess_order_items(order_items)
    products = preprocess_products(products)
    categories = preprocess_categories(categories)
    reviews = preprocess_reviews(reviews)

    print("✅ Preprocessing Completed!\n")

    # =========================
    # 3. Merge Data
    # =========================
    df = merge_all(customers, orders, order_items, products, categories, reviews)
    print("✅ Data Merged!\n")

    # =========================
    # 4. Outlier Removal
    # =========================
    df = clean_outliers_with_boxplots(df)
    print("✅ Outliers Removed!\n")

    # =========================
    # 5. Save Clean Data
    # =========================
    save_raw_processed_data(df, "processed_data/Raw_Processed_Data.csv")
    print("✅ Cleaned data saved successfully!\n")



# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    main()