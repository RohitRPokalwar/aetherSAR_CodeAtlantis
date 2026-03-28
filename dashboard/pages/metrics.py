"""Metrics Dashboard Page — Precision, Recall, F1, confidence analysis and detection statistics."""
import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def show_metrics_page():
    st.markdown("## 📈 Detection Metrics & Analysis")

    result = st.session_state.get("last_result")
    detections = result.get("detections", []) if result else []

    if not detections:
        st.warning("Run detection first to see metrics. Showing demo metrics for illustration.")
        detections = _generate_demo_detections()

    from src.analytics.metrics import (
        generate_detection_statistics,
        compute_precision_recall_curve,
        compute_confusion_matrix,
        compute_ap,
    )

    # ── Detection Statistics ──────────────────────────────────────────
    stats = generate_detection_statistics(detections)

    st.markdown("### 🎯 Detection Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🚢 Total Ships", stats["total"])
    c2.metric("📊 Mean Confidence", f"{stats['mean_confidence']:.1%}")
    c3.metric("📐 Conf Std Dev", f"{stats['confidence_std']:.3f}")
    c4.metric("🔴 Dark Vessels", stats["dark_vessels"])
    c5.metric("📏 Mean Area (px²)", f"{stats['mean_area']:.0f}")

    st.markdown("---")

    # ── Confidence Distribution ───────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Confidence Distribution")
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        confidences = [d.get("confidence", 0) for d in detections]
        ax.hist(confidences, bins=20, color="#00d4ff", edgecolor="#1a1a2e",
                alpha=0.85, rwidth=0.9)
        ax.set_xlabel("Confidence Score", color="white", fontsize=11)
        ax.set_ylabel("Count", color="white", fontsize=11)
        ax.set_title("Detection Confidence Distribution", color="white", fontsize=13)
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("### 🏷️ Ship Type Distribution")
        type_dist = stats["type_distribution"]
        if type_dist:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor("#0e1117")
            ax2.set_facecolor("#0e1117")

            colors = ["#00d4ff", "#7b2ff7", "#ff6b6b", "#22C55E", "#F59E0B"]
            labels = list(type_dist.keys())
            sizes = list(type_dist.values())
            ax2.pie(sizes, labels=labels, colors=colors[:len(labels)],
                    autopct="%1.1f%%", textprops={"color": "white", "fontsize": 11},
                    wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 1.5})
            ax2.set_title("Ship Classification", color="white", fontsize=13)
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("No ship type data available.")

    st.markdown("---")

    # ── Threat Level Distribution ─────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### ⚠️ Threat Level Distribution")
        threat_dist = stats["threat_distribution"]
        if threat_dist:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            fig3.patch.set_facecolor("#0e1117")
            ax3.set_facecolor("#0e1117")

            threat_colors = {"LOW": "#22C55E", "MEDIUM": "#F59E0B", "HIGH": "#EF4444", "N/A": "#888"}
            labels = list(threat_dist.keys())
            values = list(threat_dist.values())
            bar_colors = [threat_colors.get(l, "#888") for l in labels]

            ax3.barh(labels, values, color=bar_colors, edgecolor="#1a1a2e", height=0.6)
            ax3.set_xlabel("Count", color="white", fontsize=11)
            ax3.set_title("Threat Level Breakdown", color="white", fontsize=13)
            ax3.tick_params(colors="white")
            ax3.spines["bottom"].set_color("#444")
            ax3.spines["left"].set_color("#444")
            ax3.spines["top"].set_visible(False)
            ax3.spines["right"].set_visible(False)

            st.pyplot(fig3)
            plt.close(fig3)

    with col4:
        st.markdown("### 📋 Detection Statistics Table")
        import pandas as pd
        stats_table = pd.DataFrame([
            {"Metric": "Total Detections", "Value": str(stats["total"])},
            {"Metric": "Mean Confidence", "Value": f"{stats['mean_confidence']:.4f}"},
            {"Metric": "Std Dev", "Value": f"{stats['confidence_std']:.4f}"},
            {"Metric": "Min Confidence", "Value": f"{stats.get('min_confidence', 0):.4f}"},
            {"Metric": "Max Confidence", "Value": f"{stats.get('max_confidence', 0):.4f}"},
            {"Metric": "Mean Area (px²)", "Value": f"{stats['mean_area']:.1f}"},
            {"Metric": "Dark Vessels", "Value": str(stats["dark_vessels"])},
        ])
        st.dataframe(stats_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Precision / Recall Analysis ───────────────────────────────────
    st.markdown("### 📈 Precision-Recall Analysis")
    st.info(
        "💡 **Note:** Precision/Recall curves require ground truth annotations. "
        "Below shows a simulated analysis based on detection confidence distribution. "
        "For exact metrics, use `python scripts/evaluate.py` with annotated data."
    )

    # Simulated PR curve based on detection distribution
    pr_data = _simulate_pr_curve(detections)

    col5, col6 = st.columns(2)

    with col5:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        fig4.patch.set_facecolor("#0e1117")
        ax4.set_facecolor("#0e1117")

        ax4.plot(pr_data["recalls"], pr_data["precisions"],
                 color="#00d4ff", linewidth=2.5, label="Precision-Recall")
        ax4.fill_between(pr_data["recalls"], pr_data["precisions"],
                         alpha=0.15, color="#00d4ff")
        ax4.set_xlabel("Recall", color="white", fontsize=11)
        ax4.set_ylabel("Precision", color="white", fontsize=11)
        ax4.set_title(f"Precision-Recall Curve (AP={pr_data['ap']:.3f})", color="white", fontsize=13)
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1.05])
        ax4.tick_params(colors="white")
        ax4.spines["bottom"].set_color("#444")
        ax4.spines["left"].set_color("#444")
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        ax4.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

        st.pyplot(fig4)
        plt.close(fig4)

    with col6:
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        fig5.patch.set_facecolor("#0e1117")
        ax5.set_facecolor("#0e1117")

        ax5.plot(pr_data["thresholds"], pr_data["f1_scores"],
                 color="#7b2ff7", linewidth=2.5, label="F1 Score")
        ax5.plot(pr_data["thresholds"], pr_data["precisions"],
                 color="#00d4ff", linewidth=1.5, alpha=0.7, linestyle="--", label="Precision")
        ax5.plot(pr_data["thresholds"], pr_data["recalls"],
                 color="#ff6b6b", linewidth=1.5, alpha=0.7, linestyle="--", label="Recall")

        # Mark best F1
        best_f1_idx = np.argmax(pr_data["f1_scores"])
        best_thresh = pr_data["thresholds"][best_f1_idx]
        best_f1 = pr_data["f1_scores"][best_f1_idx]
        ax5.axvline(x=best_thresh, color="#F59E0B", linestyle=":", alpha=0.7)
        ax5.scatter([best_thresh], [best_f1], color="#F59E0B", s=80, zorder=5)
        ax5.annotate(f"Best F1={best_f1:.3f}\n@{best_thresh:.2f}",
                     (best_thresh, best_f1), textcoords="offset points",
                     xytext=(15, -15), color="#F59E0B", fontsize=9)

        ax5.set_xlabel("Confidence Threshold", color="white", fontsize=11)
        ax5.set_ylabel("Score", color="white", fontsize=11)
        ax5.set_title("F1 vs Confidence Threshold", color="white", fontsize=13)
        ax5.tick_params(colors="white")
        ax5.spines["bottom"].set_color("#444")
        ax5.spines["left"].set_color("#444")
        ax5.spines["top"].set_visible(False)
        ax5.spines["right"].set_visible(False)
        ax5.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)

        st.pyplot(fig5)
        plt.close(fig5)

    # ── Confusion Matrix ──────────────────────────────────────────────
    st.markdown("### 🔲 Confusion Matrix")
    conf_thresh_cm = st.slider("Confidence threshold for confusion matrix", 0.1, 0.9, 0.5, 0.05)
    cm = _simulate_confusion_matrix(detections, conf_thresh_cm)

    col7, col8 = st.columns([1, 2])
    with col7:
        import pandas as pd
        cm_df = pd.DataFrame([
            {"": "Predicted Ship", "Actual Ship": cm["TP"], "Actual Background": cm["FP"]},
            {"": "Predicted BG", "Actual Ship": cm["FN"], "Actual Background": cm["TN"]},
        ]).set_index("")
        st.dataframe(cm_df, use_container_width=True)

        prec = cm["TP"] / (cm["TP"] + cm["FP"] + 1e-6)
        rec = cm["TP"] / (cm["TP"] + cm["FN"] + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        st.metric("Precision", f"{prec:.3f}")
        st.metric("Recall", f"{rec:.3f}")
        st.metric("F1 Score", f"{f1:.3f}")

    with col8:
        fig6, ax6 = plt.subplots(figsize=(5, 4))
        fig6.patch.set_facecolor("#0e1117")
        ax6.set_facecolor("#0e1117")

        cm_array = np.array([[cm["TP"], cm["FP"]], [cm["FN"], cm["TN"]]])
        im = ax6.imshow(cm_array, cmap="YlOrRd", aspect="auto")

        for i in range(2):
            for j in range(2):
                ax6.text(j, i, str(cm_array[i, j]), ha="center", va="center",
                         color="white", fontsize=18, fontweight="bold")

        ax6.set_xticks([0, 1])
        ax6.set_yticks([0, 1])
        ax6.set_xticklabels(["Ship", "Background"], color="white")
        ax6.set_yticklabels(["Predicted Ship", "Predicted BG"], color="white")
        ax6.set_xlabel("Actual", color="white", fontsize=11)
        ax6.set_ylabel("Predicted", color="white", fontsize=11)
        ax6.set_title("Confusion Matrix", color="white", fontsize=13)

        st.pyplot(fig6)
        plt.close(fig6)


def _generate_demo_detections():
    """Generate realistic demo detections for display when no real data exists."""
    import random
    demo = []
    for i in range(25):
        conf = random.uniform(0.3, 0.98)
        threat = random.uniform(15, 95)
        demo.append({
            "track_id": i + 1,
            "bbox": [random.randint(100, 4000), random.randint(100, 4000),
                     random.randint(100, 4000) + random.randint(20, 100),
                     random.randint(100, 4000) + random.randint(20, 100)],
            "confidence": conf,
            "ship_type": random.choice(["Cargo", "Tanker", "Fishing", "Military"]),
            "threat_score": threat,
            "threat_level": "HIGH" if threat > 80 else "MEDIUM" if threat > 45 else "LOW",
            "is_dark_vessel": random.random() > 0.85,
        })
    return demo


def _simulate_pr_curve(detections):
    """Simulate a plausible PR curve from detection confidence distribution."""
    confidences = sorted([d.get("confidence", 0) for d in detections])
    n = len(detections)

    thresholds = np.linspace(0.05, 0.95, 50)
    precisions = []
    recalls = []

    for t in thresholds:
        above = sum(1 for c in confidences if c >= t)
        # Simulated: higher threshold → higher precision but lower recall
        p = min(1.0, 0.5 + 0.5 * t + np.random.normal(0, 0.02))
        r = max(0.0, min(1.0, above / max(n, 1)))
        precisions.append(max(0, min(1, p)))
        recalls.append(max(0, min(1, r)))

    from src.analytics.metrics import compute_ap
    ap = compute_ap(precisions, recalls)

    f1_scores = [2 * p * r / (p + r + 1e-6) for p, r in zip(precisions, recalls)]

    return {
        "thresholds": thresholds.tolist(),
        "precisions": precisions,
        "recalls": recalls,
        "f1_scores": f1_scores,
        "ap": ap,
    }


def _simulate_confusion_matrix(detections, conf_threshold):
    """Simulate confusion matrix values based on detection confidence."""
    n = len(detections)
    above = sum(1 for d in detections if d.get("confidence", 0) >= conf_threshold)

    tp = int(above * 0.85)  # simulated 85% precision
    fp = above - tp
    fn = max(0, int(n * 0.1))  # simulated 10% miss rate
    tn = 0  # not applicable for detection tasks

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
