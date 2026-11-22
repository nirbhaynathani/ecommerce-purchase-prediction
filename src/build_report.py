"""Build the polished PDF project report from the saved model artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
OUTPUT_PDF = ROOT / "reports" / "Ecommerce_Customer_Purchase_Prediction_Report.pdf"
TEAL = colors.HexColor("#167D8D")
NAVY = colors.HexColor("#153B52")
PALE = colors.HexColor("#EAF3F5")
GRAY = colors.HexColor("#53636E")


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25, leading=30, textColor=NAVY, alignment=TA_LEFT, spaceAfter=8))
    styles.add(ParagraphStyle(name="SubtitleCustom", parent=styles["Normal"], fontName="Helvetica", fontSize=12, leading=17, textColor=TEAL, spaceAfter=10))
    styles.add(ParagraphStyle(name="H1Custom", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=NAVY, spaceBefore=4, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2Custom", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=TEAL, spaceBefore=5, spaceAfter=5))
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=9, leading=13, textColor=colors.HexColor("#24323B"), spaceAfter=6))
    styles.add(ParagraphStyle(name="SmallCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.8, leading=10.5, textColor=GRAY, spaceAfter=4))
    styles.add(ParagraphStyle(name="CalloutCustom", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=NAVY, backColor=PALE, borderColor=TEAL, borderWidth=0.8, borderPadding=8, spaceBefore=5, spaceAfter=8))
    return styles


def paragraph(text, style):
    return Paragraph(text, style)


def figure(path: Path, width: float, height: float):
    return Image(str(path), width=width, height=height, kind="proportional")


def make_table(rows, widths, header=True, font_size=8):
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DEE3")),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#24323B")),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8F9")]),
        ])
    table.setStyle(TableStyle(commands))
    return table


def decorate(canvas, doc):
    canvas.saveState()
    width, height = A4
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#D5DEE3"))
        canvas.line(0.65 * inch, height - 0.52 * inch, width - 0.65 * inch, height - 0.52 * inch)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(TEAL)
        canvas.drawString(0.65 * inch, height - 0.40 * inch, "SHOPSMART  |  CUSTOMER PURCHASE PREDICTION")
    canvas.setStrokeColor(colors.HexColor("#D5DEE3"))
    canvas.line(0.65 * inch, 0.48 * inch, width - 0.65 * inch, 0.48 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(0.65 * inch, 0.31 * inch, "Sapphire IQ ML Project  |  Prepared 23 November 2026")
    canvas.drawRightString(width - 0.65 * inch, 0.31 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_report():
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    data = pd.read_csv(ROOT / "data" / "Ecommerce_Purchase_Prediction_DataSet.csv")
    predictions = pd.read_csv(RESULTS / "sample_new_customer_predictions.csv")
    s = make_styles()
    story = []
    m = metrics["metrics"]

    story += [
        Spacer(1, 0.28 * inch),
        paragraph("E-Commerce Customer<br/>Purchase Prediction", s["CoverTitle"]),
        paragraph("ShopSmart  |  Supervised learning  |  Logistic Regression", s["SubtitleCustom"]),
        paragraph("PROJECT REPORT  •  SAPPHIRE IQ MACHINE LEARNING PROJECT", s["SmallCustom"]),
        Spacer(1, 0.08 * inch),
        paragraph("Executive summary", s["H1Custom"]),
        paragraph(
            "This project builds a binary classification prototype to estimate whether a customer session ends in a purchase. "
            "The supplied 10,000-row dataset was explored and prepared using a leakage-safe scikit-learn pipeline. A Logistic Regression model was evaluated on a stratified holdout set of 2,000 sessions.",
            s["BodyCustom"],
        ),
        paragraph(
            f"On the holdout set, the model achieved <b>{m['accuracy']:.1%} accuracy</b>, "
            f"<b>{m['f1_score']:.1%} F1-score</b>, and <b>{m['roc_auc']:.3f} ROC-AUC</b>. "
            f"Recall was {m['recall']:.1%}; the 0.50 threshold identifies many purchasers, while also flagging non-purchasers.",
            s["CalloutCustom"],
        ),
        paragraph("Holdout performance", s["H2Custom"]),
        make_table(
            [["Metric", "Result"], ["Accuracy", f"{m['accuracy']:.2%}"], ["Precision", f"{m['precision']:.2%}"], ["Recall", f"{m['recall']:.2%}"], ["F1-score", f"{m['f1_score']:.2%}"], ["ROC-AUC", f"{m['roc_auc']:.3f}"], ["Majority-class accuracy baseline", f"{m['majority_class_baseline_accuracy']:.2%}"]],
            [3.9 * inch, 1.35 * inch],
        ),
        Spacer(1, 0.12 * inch),
        paragraph("Project question and target", s["H2Custom"]),
        paragraph(
            "Can customer and browsing information predict whether a session results in a purchase? The target is <b>Purchase</b>: "
            "0 means no purchase and 1 means purchase. The model is intended to help marketing teams prioritize likely buyers for review; it is not a causal analysis or a production deployment.",
            s["BodyCustom"],
        ),
        paragraph(f"Dataset: {metrics['dataset_rows']:,} customer sessions, {metrics['predictor_count']} predictors, {metrics['train_rows']:,} training rows and {metrics['test_rows']:,} test rows.", s["SmallCustom"]),
        paragraph("Predictors used: Age, Gender, Device_Type, Traffic_Source, Membership_Type, Payment_Method, Session_Duration_Min, Pages_Viewed, Products_Viewed, Cart_Items, Cart_Value_INR, Discount_Percent, Previous_Orders, Days_Since_Last_Visit, Email_Open_Rate, Customer_Rating, and Returns_Last_Year.", s["SmallCustom"]),
        PageBreak(),
    ]

    story += [
        paragraph("Data understanding and preparation", s["H1Custom"]),
        paragraph(
            f"The source contains {len(data):,} records and {data.shape[1]} fields. Purchase class 1 represents "
            f"{metrics['purchase_rate']:.1%} of records ({metrics['target_distribution']['1']:,} sessions); class 0 represents "
            f"{metrics['target_distribution']['0']:,}. There are {metrics['missing_cells']:,} missing predictor cells and "
            f"{metrics['duplicate_rows']} fully duplicated rows. Customer_ID is not used as a model feature.",
            s["BodyCustom"],
        ),
        paragraph("Outcome mix and missingness", s["H2Custom"]),
        Table([[figure(FIGURES / "purchase_distribution.png", 3.05 * inch, 2.0 * inch), figure(FIGURES / "missing_values.png", 3.05 * inch, 2.0 * inch)]], colWidths=[3.2 * inch, 3.2 * inch], style=TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 3)])),
        paragraph("Preprocessing decisions", s["H2Custom"]),
        make_table(
            [["Step", "Implementation"],
             ["Missing numeric values", "Median imputation fitted on the training split"],
             ["Outliers", "Numeric values capped at training-split 1.5 × IQR fences; records are retained"],
             ["Feature scaling", "StandardScaler applied to numeric predictors"],
             ["Categorical data", "Most-frequent imputation followed by one-hot encoding"],
             ["Leakage control", "All transforms are inside the training pipeline; Customer_ID is excluded"]],
            [1.55 * inch, 4.85 * inch], font_size=7.5,
        ),
        Spacer(1, 0.05 * inch),
        paragraph("Observed patterns", s["H2Custom"]),
        Table([[figure(FIGURES / "purchase_rate_by_membership.png", 3.0 * inch, 2.0 * inch), figure(FIGURES / "cart_value_by_purchase.png", 3.0 * inch, 2.0 * inch)]], colWidths=[3.2 * inch, 3.2 * inch], style=TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 3)])),
        paragraph("Observed purchase rates were about 72.1% for Premium members, 66.1% for Regular members, and 61.0% for New members. These group comparisons describe this dataset and do not show that membership causes purchasing.", s["SmallCustom"]),
        PageBreak(),
    ]

    cm = metrics["confusion_matrix_labels_0_then_1"]
    story += [
        paragraph("Model and evaluation", s["H1Custom"]),
        paragraph(
            "A Logistic Regression classifier was fit using a stratified 80/20 train/test split and random seed 42. "
            "The model uses the 0.50 probability threshold. Accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC were calculated from the held-out test data.",
            s["BodyCustom"],
        ),
        paragraph("Confusion matrix", s["H2Custom"]),
        make_table([["Actual \\ Predicted", "No purchase (0)", "Purchase (1)"], ["No purchase (0)", f"{cm[0][0]:,} true negatives", f"{cm[0][1]:,} false positives"], ["Purchase (1)", f"{cm[1][0]:,} false negatives", f"{cm[1][1]:,} true positives"]], [2.1 * inch, 2.15 * inch, 2.15 * inch]),
        Spacer(1, 0.08 * inch),
        figure(FIGURES / "model_evaluation.png", 6.2 * inch, 2.9 * inch),
        paragraph("The model finds 87.1% of purchasers in this test split (recall), while 72.6% of sessions predicted as purchases are actual purchasers (precision). The 426 false positives may create wasted outreach; 167 purchasers are missed. Threshold selection should reflect campaign capacity and the relative business cost of each error.", s["BodyCustom"]),
        paragraph("Largest coefficient signals", s["H2Custom"]),
        figure(FIGURES / "largest_coefficients.png", 5.8 * inch, 3.55 * inch),
        paragraph("Coefficients show model associations after preprocessing. One-hot category coefficients are relative to the encoder's omitted reference category. Coefficients are not causal effects and can be affected by correlated predictors.", s["SmallCustom"]),
        PageBreak(),
    ]

    rows = [["Membership", "Device", "Session (min)", "Cart items", "Cart value (INR)", "Purchase probability", "Predicted class"]]
    for _, row in predictions.iterrows():
        rows.append([
            str(row["Membership_Type"]), str(row["Device_Type"]), f"{row['Session_Duration_Min']:.1f}",
            str(int(row["Cart_Items"])), f"{row['Cart_Value_INR']:,.0f}",
            f"{row['purchase_probability']:.1%}", "Purchase" if row["predicted_purchase"] == 1 else "No purchase",
        ])
    story += [
        paragraph("Example use and limitations", s["H1Custom"]),
        paragraph("Example predictions for new customer profiles", s["H2Custom"]),
        paragraph("These three illustrative profiles show the probability and class returned by the fitted pipeline. A probability is a model score, not a guarantee of customer behavior.", s["BodyCustom"]),
        make_table(rows, [0.83 * inch, 0.74 * inch, 0.76 * inch, 0.56 * inch, 0.9 * inch, 1.03 * inch, 1.0 * inch], font_size=6.8),
        Spacer(1, 0.12 * inch),
        paragraph("Recommended decision-support workflow", s["H2Custom"]),
        paragraph("1. Score eligible sessions with the saved preprocessing/model pipeline.<br/>2. Rank potential buyers by predicted purchase probability, then apply campaign capacity and business rules.<br/>3. Measure campaign outcomes with a controlled experiment before claiming conversion improvement.<br/>4. Monitor data quality and model performance; retrain only after checking that future data is suitable.", s["BodyCustom"]),
        paragraph("Limitations", s["H2Custom"]),
        paragraph("This report evaluates one random holdout split, not a later time period or an external customer population. The supplied brief does not describe the dataset's sampling or generation process. The threshold is fixed at 0.50 rather than optimized for a known campaign cost. Purchase probabilities may require calibration before operational use. Dataset associations, model coefficients, and predicted probabilities do not establish causality. Validate performance, privacy requirements, and fairness across relevant customer groups before any real deployment.", s["BodyCustom"]),
        paragraph("Reproducibility", s["H2Custom"]),
        paragraph("The repository includes the raw supplied CSV, the training and evaluation module, an explanatory notebook, a saved scikit-learn pipeline, metrics, holdout predictions, new-customer examples, and the figures used in this report. Run `python -m src.modeling` from the project folder to regenerate model artifacts; the PDF can then be recreated with `python -m src.build_report` in an environment with ReportLab installed.", s["BodyCustom"]),
        paragraph("Conclusion", s["H2Custom"]),
        paragraph("The requested Logistic Regression classification workflow has been completed. Its holdout performance is better than the majority-class accuracy baseline in this single split and it demonstrates how session and customer attributes can support buyer prioritization. Further temporal validation and cost-aware threshold setting are needed before business use.", s["CalloutCustom"]),
    ]

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF), pagesize=A4, rightMargin=0.65 * inch, leftMargin=0.65 * inch,
        topMargin=0.7 * inch, bottomMargin=0.65 * inch,
        title="E-Commerce Customer Purchase Prediction Report",
        author="ShopSmart Project Team",
    )
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    build_report()
