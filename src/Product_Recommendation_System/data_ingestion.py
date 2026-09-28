import pandas as pd

def load_all_datasets():
    """
    load all six olist datasets using fixed file paths
    Return six pandas DataFrames
    """
    try:
        customers= pd.read_csv('D:/FinalYearProject/olist ecommerce dataset/olist_customers_dataset.csv')
        orders= pd.read_csv("D:/FinalYearProject/olist ecommerce dataset/olist_orders_dataset.csv")
        order_items=pd.read_csv("D:/FinalYearProject/olist ecommerce dataset/olist_order_items_dataset.csv")
        products=pd.read_csv("D:/FinalYearProject/olist ecommerce dataset/olist_products_dataset.csv")
        categories=pd.read_csv("D:/FinalYearProject/olist ecommerce dataset/product_category_name_translation.csv")
        reviews=pd.read_csv("D:/FinalYearProject/olist ecommerce dataset/olist_order_reviews_dataset.csv")

        print("all datasets loaded successfully")
        print(f"Customers: {customers.shape}")
        print(f"Orders: {orders.shape}")
        print(f"Order Items: {order_items.shape}")
        print(f"Products: {products.shape}")
        print(f"Categories: {categories.shape}")
        print(f"Reviews: {reviews.shape}")
        return customers, orders, order_items, products, categories, reviews

    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None, None, None, None, None
# ------------------------
# Run the script directly
# ------------------------
if __name__ == "__main__":
    load_all_datasets()
    