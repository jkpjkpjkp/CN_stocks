import rqdatac as rq
rq.init("+861008712021", "finlab@pku100871", "222.29.71.3:16010")

import os
from datetime import datetime

# Initialize rqdatac
# Replace 'your_username' and 'your_password' with your actual Ricequant credentials
# rq.init('your_username', 'your_password')
# If you are already in a research environment that auto-initializes, you can skip explicit init.
# rq.init() 

save_dir = '.data_downloads'
def download_data():
    print("Fetching instrument list...")
    # 1. Get all instruments for the CN market
    # specific types can be requested, but to get "ALL" excluding one, we fetch all first.
    all_inst = rq.all_instruments(market='cn')
    
    # 2. Filter out stocks
    # 'CS' stands for Common Stock. We keep everything that is NOT 'CS'.
    # This effectively excludes stocks from SHG (Shanghai) and SHE (Shenzhen).
    non_stock_inst = all_inst[all_inst['type'] != 'CS']
    
    print(f"Found {len(non_stock_inst)} non-stock instruments.")
    
    # Create a directory to save the data
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 3. Iterate and download data for each instrument
    # Note: This list might include thousands of expired options and futures.
    # You might want to further filter by 'status'=='Active' if you only want currently listed ones.
    
    for index, row in non_stock_inst.iterrows():
        order_book_id = row['order_book_id']
        inst_type = row['type']
        listed_date = row['listed_date']
        
        # Determine start date
        # Some continuous contracts have '0000-00-00' as listed_date
        if listed_date == '0000-00-00':
            start_date = '2010-01-01' 
        else:
            start_date = listed_date
            
        end_date = datetime.now().date()
        
        print(f"Downloading {inst_type} - {order_book_id} ...")
        
        try:
            # Download daily data (frequency='1d')
            # You can change frequency to '1m' for minute data, but it will be significantly larger/slower
            df = rq.get_price(order_book_id, start_date=start_date, end_date=end_date, frequency='1d')
            
            if df is not None and not df.empty:
                # Save to CSV
                file_path = os.path.join(save_dir, f"{order_book_id}.csv")
                df.to_csv(file_path)
            else:
                print(f"  No data found for {order_book_id}")
                
        except Exception as e:
            print(f"  Error downloading {order_book_id}: {e}")

if __name__ == "__main__":
    # Ensure rq.init is called before running this
    # rq.init(username='...', password='...')
    download_data()