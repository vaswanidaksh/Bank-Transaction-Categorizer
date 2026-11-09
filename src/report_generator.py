"""
PDF Financial Spending Report Generator
---------------------------------------
Uses ReportLab to compile categorized bank transactions into an executive,
print-ready personal/business expense breakdown report.
"""

import io
from typing import Optional
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_pdf_report(
    df: pd.DataFrame,
    output_path: Optional[str] = None
) -> bytes:
    """
    Generates a PDF expense summary report from a categorized transactions DataFrame.
    Returns bytes, and optionally saves to output_path if provided.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        output_path if output_path else buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=15
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=12,
        spaceAfter=8
    )

    # Document Header
    story.append(Paragraph("🏦 Bank Transaction Categorizer — Spending Report", title_style))
    story.append(Paragraph("Automated Financial Breakdown & Category Intelligence", subtitle_style))
    story.append(Spacer(1, 10))

    # Calculations
    has_amount = "amount" in df.columns
    has_type = "transaction_type" in df.columns

    total_txns = len(df)
    if has_amount and has_type:
        total_credit = df[df["transaction_type"] == "CREDIT"]["amount"].sum()
        total_debit = df[df["transaction_type"] == "DEBIT"]["amount"].sum()
    elif has_amount:
        total_credit = 0.0
        total_debit = df["amount"].sum()
    else:
        total_credit = 0.0
        total_debit = 0.0

    net_savings = total_credit - total_debit

    # KPI Summary Box
    kpi_data = [
        ["Total Transactions", "Total Income (Credits)", "Total Spending (Debits)", "Net Savings / Balance"],
        [
            f"{total_txns:,}",
            f"INR {total_credit:,.2f}",
            f"INR {total_debit:,.2f}",
            f"INR {net_savings:,.2f}"
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[130, 140, 140, 130])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 15))

    # Category Breakdown Table
    story.append(Paragraph("Category-Wise Spending Breakdown", h2_style))
    category_col = "predicted_category" if "predicted_category" in df.columns else "category"

    if category_col in df.columns and has_amount:
        debit_df = df[df["transaction_type"] == "DEBIT"] if has_type else df
        cat_summary = debit_df.groupby(category_col).agg(
            Count=("amount", "count"),
            Total_Spend=("amount", "sum")
        ).reset_index()

        cat_summary["Percentage"] = (
            (cat_summary["Total_Spend"] / total_debit * 100).round(1) if total_debit > 0 else 0
        )
        cat_summary = cat_summary.sort_values(by="Total_Spend", ascending=False)

        cat_table_data = [["Category", "Txn Count", "Total Amount (INR)", "Share of Spend"]]
        for _, row in cat_summary.iterrows():
            cat_table_data.append([
                str(row[category_col]),
                str(int(row["Count"])),
                f"INR {row['Total_Spend']:,.2f}",
                f"{row['Percentage']}%"
            ])

        cat_table = Table(cat_table_data, colWidths=[180, 80, 150, 130])
        cat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ]))
        story.append(cat_table)
    story.append(Spacer(1, 15))

    # Top Recent Transactions Sample
    story.append(Paragraph("Sample Categorized Entries", h2_style))
    sample_df = df.head(8)
    txn_rows = [["Date", "Description", "Category", "Amount"]]
    for _, r in sample_df.iterrows():
        desc = str(r.get("raw_description", ""))[:32]
        cat = str(r.get(category_col, ""))
        amt = f"INR {r.get('amount', 0):,.2f}" if has_amount else "-"
        dt = str(r.get("date", "-"))
        txn_rows.append([dt, desc, cat, amt])

    txn_table = Table(txn_rows, colWidths=[80, 220, 140, 100])
    txn_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#475569")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(txn_table)

    doc.build(story)

    if output_path:
        with open(output_path, "rb") as f:
            return f.read()
    else:
        buffer.seek(0)
        return buffer.getvalue()


if __name__ == "__main__":
    test_df = pd.read_csv("data/sample/sample_statement.csv")
    pdf_bytes = generate_pdf_report(test_df, output_path="data/sample/test_report.pdf")
    print(f"Generated test PDF report ({len(pdf_bytes)} bytes) -> data/sample/test_report.pdf")

