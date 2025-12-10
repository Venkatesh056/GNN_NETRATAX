import pandas as pd
from datetime import datetime
from graph_builder import build_pyg_data


def _write(tmp_path, name, df):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def test_shapes_and_features(tmp_path):
    companies = pd.DataFrame(
        {
            "company_id": ["C1", "C2", "C3"],
            "registration_date": ["2020-01-01", "2021-06-15", "2022-03-10"],
        }
    )
    invoices = pd.DataFrame(
        {
            "invoice_id": ["I1", "I2"],
            "seller_id": ["C1", "C2"],
            "buyer_id": ["C2", "C3"],
            "amount": [100.0, 200.0],
            "invoice_date": ["2024-01-01", "2024-06-01"],
        }
    )
    relations = pd.DataFrame(
        {
            "src_company_id": ["C1"],
            "dst_company_id": ["C3"],
            "relation_type": ["ownership"],
        }
    )

    c_path = _write(tmp_path, "companies.csv", companies)
    i_path = _write(tmp_path, "invoices.csv", invoices)
    r_path = _write(tmp_path, "relations.csv", relations)

    hetero = build_pyg_data(c_path, i_path, r_path, now=datetime(2024, 7, 1))

    # Node feature shapes
    assert hetero["company"].x.shape == (3, 6)
    assert hetero["invoice"].x.shape == (2, 2)

    # Edge blocks and attrs
    assert hetero["company", "transacts", "invoice"].edge_index.shape[1] == 2
    assert hetero["invoice", "billed_to", "company"].edge_index.shape[1] == 2
    assert hetero["company", "related", "company"].edge_index.shape[1] == 1

    assert hetero["company", "transacts", "invoice"].edge_attr.shape == (2, 3)
    assert hetero["company", "related", "company"].edge_attr.shape == (1, 3)

    # Feature ordering: degree, avg_invoice_amount, transaction_count, pagerank, betweenness, registration_age_days
    company_feats = hetero["company"].x.numpy()
    assert company_feats[0, 0] >= 0  # degree
    assert company_feats[0, 1] >= 0  # avg invoice amount
    assert company_feats[0, 2] >= 1  # transaction count for C1
    assert company_feats[0, 5] > 0   # registration age days for C1

    # Invoice features: amount and age_days
    invoice_feats = hetero["invoice"].x.numpy()
    assert invoice_feats[0, 0] == 100.0
    assert invoice_feats[0, 1] > 0


def test_registration_age_is_deterministic(tmp_path):
    companies = pd.DataFrame({"company_id": ["C1"], "registration_date": ["2020-01-01"]})
    invoices = pd.DataFrame(
        {
            "invoice_id": ["I1"],
            "seller_id": ["C1"],
            "buyer_id": ["C1"],
            "amount": [50.0],
            "invoice_date": ["2024-01-01"],
        }
    )
    relations = pd.DataFrame({"src_company_id": ["C1"], "dst_company_id": ["C1"], "relation_type": ["ownership"]})

    c_path = _write(tmp_path, "companies.csv", companies)
    i_path = _write(tmp_path, "invoices.csv", invoices)
    r_path = _write(tmp_path, "relations.csv", relations)

    hetero = build_pyg_data(c_path, i_path, r_path, now=datetime(2024, 1, 2))
    # registration_age_days should be exactly 1462 days between 2024-01-02 and 2020-01-01
    assert hetero["company"].x.shape == (1, 6)
    assert hetero["company"].x[0, 5].item() == 1462
