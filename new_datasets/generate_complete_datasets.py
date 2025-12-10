"""
Generate 10 complete datasets for incremental learning.
Each dataset is a standalone CSV with all company info + engineered features ready for upload.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import random
import string
from datetime import datetime, timedelta

root = Path(__file__).parent
rng = random.Random(42)
np.random.seed(42)

# Constants
N_DATASETS = 10
N_COMPANIES_PER_DATASET = 1000
N_INVOICES_PER_DATASET = 4000

industries = ['Manufacturing','Retail','Logistics','IT Services','Pharma','Automotive','Construction','Textiles','Chemicals','Food & Bev','Energy','Metals','Telecom','Consulting']
cities = ['Mumbai','Delhi','Bengaluru','Hyderabad','Chennai','Pune','Ahmedabad','Kolkata','Jaipur','Surat']
states = ['MH','DL','KA','TS','TN','MH','GJ','WB','RJ','GJ']
streets = ['MG Road','Ring Rd','Outer Ring','Industrial Area','Tech Park','Main St','Market Rd','Station Rd']

def rand_pan():
    letters = ''.join(rng.choice(string.ascii_uppercase) for _ in range(5))
    digits = ''.join(rng.choice(string.digits) for _ in range(4))
    last = rng.choice(string.ascii_uppercase)
    return letters + digits + last

def rand_gstin(state, pan):
    return f"{state}{pan}{rng.randint(1000,9999)}"

def rand_address():
    return f"{rng.randint(1,999)} {rng.choice(streets)}, {rng.choice(cities)}"

def rand_company_id(dataset_num):
    return f"D{dataset_num:02d}C{rng.randint(1000,9999)}"

def rand_invoice_id(dataset_num):
    return f"D{dataset_num:02d}I{rng.randint(10000,99999)}"

# Generate 10 datasets
for dataset_num in range(1, N_DATASETS + 1):
    print(f"\n=== Generating Dataset {dataset_num} ===")
    
    # Reset RNG per dataset for consistency but different data
    rng_local = random.Random(42 + dataset_num)
    np.random.seed(42 + dataset_num)
    
    companies = []
    invoices = []
    
    # Generate companies
    company_ids = []
    for i in range(N_COMPANIES_PER_DATASET):
        pan = rand_pan()
        state = rng_local.choice(states)
        gstin = rand_gstin(state, pan)
        company_id = f"D{dataset_num:02d}C{rng_local.randint(1000,9999)}"
        company_ids.append(company_id)
        
        registration_date = datetime(2018, 1, 1) + timedelta(days=rng_local.randint(0, 6*365))
        turnover = max(100000, np.random.lognormal(mean=12, sigma=0.7))
        
        # Fraud pattern flags
        is_shell = 1 if rng_local.random() < 0.08 else 0
        is_high_risk = 1 if rng_local.random() < 0.06 else 0
        is_fraud = max(is_shell, is_high_risk)
        
        companies.append({
            'company_id': company_id,
            'name': f"{rng_local.choice(['Acme','Global','Tech','Prime','Star'])} {i:03d}",
            'GSTIN': gstin,
            'PAN': pan,
            'registration_date': registration_date.date().isoformat(),
            'address': rand_address(),
            'city': rng_local.choice(cities),
            'state': state,
            'industry': rng_local.choice(industries),
            'avg_monthly_turnover': round(float(turnover), 2),
            'is_shell': is_shell,
            'is_high_risk': is_high_risk,
            'is_fraud': is_fraud
        })
    
    companies_df = pd.DataFrame(companies)
    
    # Generate invoices with features
    for _ in range(N_INVOICES_PER_DATASET):
        seller, buyer = rng_local.sample(company_ids, 2)
        invoice_date = datetime(2024, 1, 1) + timedelta(days=rng_local.randint(0, 180))
        
        seller_fraud = companies_df[companies_df['company_id'] == seller]['is_fraud'].values[0]
        if seller_fraud:
            amount = rng_local.uniform(150000, 800000)
        else:
            amount = max(500, np.random.lognormal(mean=10, sigma=0.8))
        
        itc_claimed = amount * rng_local.uniform(0.05, 0.18)
        gst_amount = amount * rng_local.uniform(0.05, 0.18)
        
        hsn_codes = [f"{rng_local.randint(1000, 9999)}" for _ in range(rng_local.randint(1, 3))]
        invoice_items = ";".join(hsn_codes)
        
        invoices.append({
            'invoice_id': f"D{dataset_num:02d}I{rng_local.randint(10000,99999)}",
            'seller_id': seller,
            'buyer_id': buyer,
            'invoice_date': invoice_date.date().isoformat(),
            'amount': round(float(amount), 2),
            'gst_amount': round(float(gst_amount), 2),
            'itc_claimed': round(float(itc_claimed), 2),
            'invoice_items': invoice_items,
            'status': rng_local.choice(['Paid', 'Pending', 'Cancelled']),
            'payment_method': rng_local.choice(['Bank Transfer', 'Cheque', 'Cash', 'UPI'])
        })
    
    invoices_df = pd.DataFrame(invoices)
    
    # Engineer company-level features from invoices
    sent_stats = invoices_df.groupby('seller_id').agg({
        'amount': ['sum', 'count', 'mean', 'std'],
        'itc_claimed': 'sum',
        'gst_amount': 'sum'
    }).fillna(0)
    sent_stats.columns = ['total_sent_amount', 'sent_invoice_count', 'avg_sent_amount', 'std_sent_amount', 'total_itc_sent', 'total_gst_sent']
    sent_stats['seller_id'] = sent_stats.index
    sent_stats = sent_stats.reset_index(drop=True)
    
    received_stats = invoices_df.groupby('buyer_id').agg({
        'amount': ['sum', 'count', 'mean', 'std'],
        'itc_claimed': 'sum',
        'gst_amount': 'sum'
    }).fillna(0)
    received_stats.columns = ['total_received_amount', 'received_invoice_count', 'avg_received_amount', 'std_received_amount', 'total_itc_received', 'total_gst_received']
    received_stats['buyer_id'] = received_stats.index
    received_stats = received_stats.reset_index(drop=True)
    
    # Merge features back to companies
    companies_df = companies_df.merge(sent_stats.rename(columns={'seller_id': 'company_id'}), on='company_id', how='left')
    companies_df = companies_df.merge(received_stats.rename(columns={'buyer_id': 'company_id'}), on='company_id', how='left')
    companies_df = companies_df.fillna(0)
    
    # Compute derived features
    companies_df['total_invoices'] = companies_df['sent_invoice_count'] + companies_df['received_invoice_count']
    companies_df['total_amount_all'] = companies_df['total_sent_amount'] + companies_df['total_received_amount']
    companies_df['avg_invoice_amount'] = np.where(
        companies_df['total_invoices'] > 0,
        companies_df['total_amount_all'] / companies_df['total_invoices'],
        0
    )
    companies_df['total_itc_all'] = companies_df['total_itc_sent'] + companies_df['total_itc_received']
    companies_df['total_gst_all'] = companies_df['total_gst_sent'] + companies_df['total_gst_received']
    companies_df['itc_ratio'] = np.where(
        companies_df['total_amount_all'] > 0,
        companies_df['total_itc_all'] / companies_df['total_amount_all'],
        0
    )
    
    # Save merged dataset
    output_file = root / f"dataset_{dataset_num:02d}_companies_complete.csv"
    companies_df.to_csv(output_file, index=False)
    print(f"   ✓ Saved {output_file.name}: {len(companies_df)} companies with {len(companies_df.columns)} features")
    
    # Save invoices for reference
    invoices_file = root / f"dataset_{dataset_num:02d}_invoices.csv"
    invoices_df.to_csv(invoices_file, index=False)
    print(f"   ✓ Saved {invoices_file.name}: {len(invoices_df)} invoices")

print("\n✅ Generated 10 complete datasets!")
print("\nEach dataset includes:")
print("  - dataset_NN_companies_complete.csv: 150 companies + all engineered features ready for upload")
print("  - dataset_NN_invoices.csv: 600 invoices (for reference)")
print("\nFeatures in companies_complete.csv:")
print("  - Identifiers: company_id, GSTIN, PAN, name")
print("  - Metadata: registration_date, address, city, state, industry, avg_monthly_turnover")
print("  - Sent (as seller): total_sent_amount, sent_invoice_count, avg_sent_amount, std_sent_amount, total_itc_sent, total_gst_sent")
print("  - Received (as buyer): total_received_amount, received_invoice_count, avg_received_amount, std_received_amount, total_itc_received, total_gst_received")
print("  - Derived: total_invoices, total_amount_all, avg_invoice_amount, total_itc_all, total_gst_all, itc_ratio")
print("  - Fraud labels: is_shell, is_high_risk, is_fraud")
print("\nUpload any dataset_NN_companies_complete.csv to the system for incremental learning.")
