"""Bundled realistic sample datasets for instant demonstration and testing.

Provides synthetic datasets covering diverse semantic types, missing values,
non-normal distributions, outliers, and realistic business relationships.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_customer_churn_sample() -> pd.DataFrame:
    """Generate a realistic Customer Churn & Demographics dataset (500 rows).

    Features:
    - CustomerID (Identifier)
    - Age (Numeric Discrete, some missing)
    - Gender (Categorical)
    - TenureMonths (Numeric)
    - ContractType (Categorical: Month-to-Month, 1-Year, 2-Year)
    - InternetService (Categorical: Fiber optic, DSL, None)
    - MonthlyCharges (Numeric Continuous with outliers)
    - TotalCharges (Numeric Continuous, correlated with Tenure & MonthlyCharges)
    - Churn (Boolean/Categorical: Yes, No)
    - SupportTickets (Numeric Discrete, Poisson)
    - SignupDate (Datetime string)
    """
    np.random.seed(42)
    n = 500

    customer_ids = [f"CUST-{1000 + i}" for i in range(n)]
    genders = np.random.choice(["Female", "Male", "Non-Binary"], size=n, p=[0.49, 0.49, 0.02])
    ages = np.random.randint(18, 75, size=n)

    # Contract types & tenure
    contract_types = np.random.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]
    )
    tenure_months = np.random.randint(1, 72, size=n)

    internet_service = np.random.choice(
        ["Fiber optic", "DSL", "No"], size=n, p=[0.45, 0.35, 0.20]
    )

    # Base monthly charges depending on internet
    base_charges = {
        "Fiber optic": np.random.normal(85, 12, size=n),
        "DSL": np.random.normal(55, 10, size=n),
        "No": np.random.normal(25, 5, size=n),
    }
    monthly_charges = np.array([
        base_charges[internet_service[i]][i] for i in range(n)
    ])
    # Add a few high outliers
    monthly_charges[np.random.choice(n, 5, replace=False)] += 80.0
    monthly_charges = np.round(np.clip(monthly_charges, 18.0, 250.0), 2)

    total_charges = np.round(monthly_charges * tenure_months * np.random.uniform(0.95, 1.05, size=n), 2)

    # Churn probability higher for month-to-month and fiber optic with high charges
    churn_prob = 0.15 + (contract_types == "Month-to-month") * 0.25 + (monthly_charges > 90) * 0.15
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    churn = np.array(["Yes" if np.random.rand() < p else "No" for p in churn_prob])

    support_tickets = np.random.poisson(lam=1.8, size=n)

    # Date generation
    base_date = pd.Timestamp("2021-01-01")
    dates = [base_date + pd.DateOffset(days=int(d)) for d in np.random.randint(0, 1000, size=n)]
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "gender": genders,
        "age": ages.astype(float),
        "tenure_months": tenure_months,
        "contract_type": contract_types,
        "internet_service": internet_service,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "support_tickets": support_tickets,
        "churn": churn,
        "signup_date": date_strs,
    })

    # Introduce realistic missing values (MCAR)
    df.loc[np.random.choice(n, 25, replace=False), "age"] = np.nan
    df.loc[np.random.choice(n, 15, replace=False), "total_charges"] = np.nan

    # Introduce a couple of duplicate rows to test duplicate detection
    df = pd.concat([df, df.iloc[[10, 25]]], ignore_index=True)

    return df


def create_employee_analytics_sample() -> pd.DataFrame:
    """Generate realistic Employee Performance & Compensation dataset (400 rows)."""
    np.random.seed(101)
    n = 400

    emp_ids = [f"EMP-{5000 + i}" for i in range(n)]
    departments = np.random.choice(
        ["Engineering", "Product", "Sales", "Marketing", "Human Resources"],
        size=n,
        p=[0.35, 0.15, 0.25, 0.15, 0.10],
    )
    education = np.random.choice(
        ["Bachelors", "Masters", "PhD", "High School"], size=n, p=[0.55, 0.30, 0.05, 0.10]
    )
    exp_years = np.round(np.random.gamma(shape=2.5, scale=3.0, size=n), 1)

    # Base salary depends on experience and department
    dept_boost = {
        "Engineering": 35000,
        "Product": 30000,
        "Sales": 20000,
        "Marketing": 15000,
        "Human Resources": 10000,
    }
    dept_boost_arr = np.array([dept_boost[d] for d in departments], dtype=float)
    salary = (
        55000.0
        + dept_boost_arr
        + (exp_years * 4200.0)
        + np.random.normal(0, 7500, size=n)
    )
    # Add high-compensation outliers
    salary[np.random.choice(n, 4, replace=False)] += 90000
    salary = np.round(salary, 0)

    perf_scores = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.15, 0.50, 0.20, 0.10])
    remote_ratio = np.random.choice([0, 50, 100], size=n, p=[0.25, 0.35, 0.40])
    attrition = np.random.choice(["Active", "Departed"], size=n, p=[0.88, 0.12])

    df = pd.DataFrame({
        "employee_id": emp_ids,
        "department": departments,
        "education_level": education,
        "experience_years": exp_years,
        "annual_salary": salary,
        "performance_score": perf_scores,
        "remote_work_pct": remote_ratio,
        "employment_status": attrition,
    })

    # Missing values
    df.loc[np.random.choice(n, 12, replace=False), "experience_years"] = np.nan
    return df


SAMPLE_DATASETS = {
    "customer_churn": {
        "name": "Customer Churn & Behavioral Analytics (502 rows)",
        "description": "Telecom customer profiles with service plans, billing metrics, support tickets, and churn flags.",
        "loader": create_customer_churn_sample,
        "filename": "customer_churn_sample.csv",
    },
    "employee_analytics": {
        "name": "Employee Talent & Compensation (400 rows)",
        "description": "Organizational HR records with departments, experience curves, salary bands, and attrition.",
        "loader": create_employee_analytics_sample,
        "filename": "employee_analytics_sample.csv",
    },
}
