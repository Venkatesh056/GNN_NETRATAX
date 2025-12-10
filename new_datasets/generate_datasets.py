import numpy as np
import pandas as pd
import random
import hashlib
import string
from datetime import datetime, timedelta
from pathlib import Path

root = Path(__file__).parent
root.mkdir(exist_ok=True)

N_DATASETS = 10
N_COMPANIES = 500
N_INVOICES = 5000
SHELL_CLUSTERS = 20
SHELL_MIN = 4
SHELL_MAX = 8
HIGH_RISK = 30

industries = [
    'Manufacturing','Retail','Logistics','IT Services','Pharma','Automotive','Construction',
    'Textiles','Chemicals','Food & Bev','Energy','Metals','Telecom','Consulting'
]
cities = ['Mumbai','Delhi','Bengaluru','Hyderabad','Chennai','Pune','Ahmedabad','Kolkata','Jaipur','Surat']
streets = ['MG Road','Ring Rd','Outer Ring','Industrial Area','Tech Park','Main St','Market Rd','Station Rd']


def rand_pan(rng):
    letters = ''.join(rng.choice(string.ascii_uppercase) for _ in range(5))
    digits = ''.join(rng.choice(string.digits) for _ in range(4))
    last = rng.choice(string.ascii_uppercase)
    return letters + digits + last


def rand_gstin(rng, pan):
    state_code = f"{rng.randint(1,35):02d}"
    suffix = ''.join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(3))
    return state_code + pan + suffix


def rand_address(rng):
    return f"{rng.randint(1,999)} {rng.choice(streets)}, {rng.choice(cities)}"


def make_name(i):
    return f"Acme_{i:03d} Pvt Ltd"


def make_hash(row):
    base = f"{row['invoice_id']}{row['seller_id']}{row['buyer_id']}{row['invoice_date']}{row['amount']}{row['invoice_items']}{row['pattern']}".encode()
    return hashlib.sha1(base).hexdigest()


def generate_dataset(ds_idx: int):
    seed = 100 + ds_idx
    rng = random.Random(seed)
    np.random.seed(seed)
    ds_path = root / f"dataset_{ds_idx:02d}"
    if ds_path.exists():
        for p in ds_path.iterdir():
            if p.is_file():
                p.unlink()
            else:
                for sub in p.rglob('*'):
                    if sub.is_file():
                        sub.unlink()
                p.rmdir()
    ds_path.mkdir(exist_ok=True)

    # Companies
    companies = []
    for i in range(N_COMPANIES):
        pan = rand_pan(rng)
        gstin = rand_gstin(rng, pan)
        registration_date = datetime(2010, 1, 1) + timedelta(days=rng.randint(0, 14 * 365))
        companies.append({
            'company_id': f'C{i:04d}',
            'name': make_name(i),
            'GSTIN': gstin,
            'PAN': pan,
            'registration_date': registration_date.date().isoformat(),
            'address': rand_address(rng),
            'industry': rng.choice(industries),
            'avg_monthly_turnover': round(max(1e5, np.random.lognormal(mean=12, sigma=0.6)), 2)
        })
    companies_df = pd.DataFrame(companies)
    companies_df['is_shell'] = 0
    companies_df['is_high_risk'] = 0

    company_ids = companies_df['company_id'].tolist()
    rng.shuffle(company_ids)

    # Shell clusters
    shell_clusters = []
    idx = 0
    for _ in range(SHELL_CLUSTERS):
        size = rng.randint(SHELL_MIN, SHELL_MAX)
        cluster_ids = company_ids[idx:idx + size]
        idx += size
        shell_clusters.append(cluster_ids)
        companies_df.loc[companies_df['company_id'].isin(cluster_ids), 'is_shell'] = 1

    # High-risk
    remaining_ids = [cid for cid in company_ids[idx:] if cid not in sum(shell_clusters, [])]
    rng.shuffle(remaining_ids)
    high_risk_ids = remaining_ids[:HIGH_RISK]
    companies_df.loc[companies_df['company_id'].isin(high_risk_ids), 'is_high_risk'] = 1

    invoices = []
    invoice_counter = 0

    # Shell invoices
    for cluster in shell_clusters:
        if len(cluster) < 2:
            continue
        for _ in range(rng.randint(8, 16)):
            a, b = rng.sample(cluster, 2)
            invoice_date = datetime(2024, 1, 1) + timedelta(days=rng.randint(0, 180))
            amount = rng.choice([50000, 75000, 100000, 125000, 150000])
            items = [f"Service_{rng.randint(1, 5)}", f"Goods_{rng.randint(1, 5)}"]
            row = {
                'invoice_id': f'INV{invoice_counter:05d}',
                'seller_id': a,
                'buyer_id': b,
                'invoice_date': invoice_date.date().isoformat(),
                'amount': float(amount),
                'itc_claimed': round(float(amount) * rng.uniform(0.12, 0.18), 2),
                'invoice_items': ';'.join(items),
                'pattern': 'shell_cluster'
            }
            row['invoice_hash'] = make_hash(row)
            invoices.append(row)
            invoice_counter += 1

    # High-risk spike invoices
    for cid in high_risk_ids:
        partners = rng.sample(company_ids, 5)
        for partner in partners:
            invoice_date = datetime(2024, 1, 1) + timedelta(days=rng.randint(0, 180))
            amount = rng.uniform(200000, 800000)
            items = [f"Bulk_{rng.randint(1, 5)}", f"HSN_{rng.randint(1000, 9999)}"]
            row = {
                'invoice_id': f'INV{invoice_counter:05d}',
                'seller_id': cid,
                'buyer_id': partner,
                'invoice_date': invoice_date.date().isoformat(),
                'amount': round(float(amount), 2),
                'itc_claimed': round(float(amount) * rng.uniform(0.12, 0.18), 2),
                'invoice_items': ';'.join(items),
                'pattern': 'spike'
            }
            row['invoice_hash'] = make_hash(row)
            invoices.append(row)
            invoice_counter += 1

    # Remaining normal invoices
    remaining = N_INVOICES - len(invoices)
    for _ in range(remaining):
        seller, buyer = rng.sample(company_ids, 2)
        invoice_date = datetime(2024, 1, 1) + timedelta(days=rng.randint(0, 180))
        amount = max(500, np.random.lognormal(mean=10, sigma=0.8))
        items = [f"Item_{rng.randint(1, 20)}", f"HSN_{rng.randint(1000, 9999)}"]
        row = {
            'invoice_id': f'INV{invoice_counter:05d}',
            'seller_id': seller,
            'buyer_id': buyer,
            'invoice_date': invoice_date.date().isoformat(),
            'amount': round(float(amount), 2),
            'itc_claimed': round(float(amount) * rng.uniform(0.08, 0.18), 2),
            'invoice_items': ';'.join(items),
            'pattern': 'normal'
        }
        row['invoice_hash'] = make_hash(row)
        invoices.append(row)
        invoice_counter += 1

    invoices_df = pd.DataFrame(invoices)

    # Relations
    relations = []
    rel_id = 0
    for cluster in shell_clusters:
        if len(cluster) < 2:
            continue
        anchor = cluster[0]
        for other in cluster[1:]:
            relations.append({
                'relation_id': f'R{rel_id:05d}',
                'src_company_id': anchor,
                'dst_company_id': other,
                'relation_type': 'shared_director',
                'strength': 0.9,
                'notes': 'Shell cluster common director'
            })
            rel_id += 1
            relations.append({
                'relation_id': f'R{rel_id:05d}',
                'src_company_id': anchor,
                'dst_company_id': other,
                'relation_type': 'shared_address',
                'strength': 0.85,
                'notes': 'Shell cluster common address'
            })
            rel_id += 1

    bank_pairs = rng.sample([(a, b) for a in company_ids for b in company_ids if a != b], 200)
    for a, b in bank_pairs:
        relations.append({
            'relation_id': f'R{rel_id:05d}',
            'src_company_id': a,
            'dst_company_id': b,
            'relation_type': 'bank_link',
            'strength': round(rng.uniform(0.3, 0.9), 2),
            'notes': 'Shared or linked bank activity'
        })
        rel_id += 1

    relations_df = pd.DataFrame(relations)

    # Derived company features
    sent = invoices_df.groupby('seller_id').agg(
        sent_invoice_count=('invoice_id', 'count'),
        total_sent_amount=('amount', 'sum'),
        total_itc_sent=('itc_claimed', 'sum')
    )
    received = invoices_df.groupby('buyer_id').agg(
        received_invoice_count=('invoice_id', 'count'),
        total_received_amount=('amount', 'sum'),
        total_itc_received=('itc_claimed', 'sum')
    )

    companies_df = companies_df.set_index('company_id')
    companies_df = companies_df.join(sent, how='left').join(received, how='left')
    companies_df.fillna(0, inplace=True)
    companies_df['total_invoices'] = companies_df['sent_invoice_count'] + companies_df['received_invoice_count']
    companies_df['total_amount_all'] = companies_df['total_sent_amount'] + companies_df['total_received_amount']
    companies_df['total_itc_all'] = companies_df['total_itc_sent'] + companies_df['total_itc_received']
    companies_df['is_fraud'] = ((companies_df['is_shell'] == 1) | (companies_df['is_high_risk'] == 1)).astype(int)
    companies_df.reset_index(inplace=True)

    # Save CSVs
    companies_df.to_csv(ds_path / 'companies.csv', index=False)
    invoices_df.to_csv(ds_path / 'invoices.csv', index=False)
    relations_df.to_csv(ds_path / 'relations.csv', index=False)

    # Samples
    companies_df.head(10).to_csv(ds_path / 'companies_sample.csv', index=False)
    invoices_df.head(10).to_csv(ds_path / 'invoices_sample.csv', index=False)
    relations_df.head(10).to_csv(ds_path / 'relations_sample.csv', index=False)

    # Excel workbook
    try:
        with pd.ExcelWriter(ds_path / 'tax_graph_dataset.xlsx') as writer:
            companies_df.to_excel(writer, sheet_name='companies', index=False)
            invoices_df.to_excel(writer, sheet_name='invoices', index=False)
            relations_df.to_excel(writer, sheet_name='relations', index=False)
            companies_df.head(10).to_excel(writer, sheet_name='companies_sample', index=False)
            invoices_df.head(10).to_excel(writer, sheet_name='invoices_sample', index=False)
            relations_df.head(10).to_excel(writer, sheet_name='relations_sample', index=False)
    except Exception as e:
        print(f"Excel export failed for dataset {ds_idx}: {e}")

    print(f"dataset_{ds_idx:02d}: companies={len(companies_df)}, invoices={len(invoices_df)}, relations={len(relations_df)}")


def write_readme():
    text = """
# Incremental Learning Synthetic Datasets (10 sets)

Schema (all datasets share columns):
- companies.csv: company_id, name, GSTIN, PAN, registration_date, address, industry, avg_monthly_turnover,
  is_shell, is_high_risk, is_fraud, sent_invoice_count, received_invoice_count, total_sent_amount,
  total_received_amount, total_invoices, total_amount_all, total_itc_sent, total_itc_received, total_itc_all
- invoices.csv: invoice_id, seller_id, buyer_id, invoice_date, amount, itc_claimed, invoice_items, pattern, invoice_hash
- relations.csv: relation_id, src_company_id, dst_company_id, relation_type (shared_director/shared_address/bank_link), strength, notes
- *_sample.csv: first 10 rows of each table
- tax_graph_dataset.xlsx: Excel workbook with all sheets above (per dataset)

Fraud signals:
- 20 shell clusters per dataset (4–8 companies each) with circular, round-amount invoices (pattern=shell_cluster, is_shell=1).
- 30 high-risk companies with spike invoices (pattern=spike, is_high_risk=1).
- is_fraud = is_shell OR is_high_risk.

Use for incremental upload:
- Upload companies.csv and invoices.csv together for the same dataset; relations.csv optional for graph enrichment.
- IDs are strings (e.g., C0001, INV00010). Amounts/ITC are numeric.
""".strip() + "\n"
    (root / 'README.md').write_text(text, encoding='utf-8')


def main():
    # clear root
    for p in root.iterdir():
        if p.is_file() and p.name != 'generate_datasets.py':
            p.unlink()
        elif p.is_dir():
            for sub in p.rglob('*'):
                if sub.is_file():
                    sub.unlink()
            for subdir in sorted([d for d in p.rglob('*') if d.is_dir()], reverse=True):
                if not any(subdir.iterdir()):
                    subdir.rmdir()
            p.rmdir()
    root.mkdir(exist_ok=True)

    for ds_idx in range(1, N_DATASETS + 1):
        generate_dataset(ds_idx)
    write_readme()
    print("Done generating all datasets.")


if __name__ == "__main__":
    main()
