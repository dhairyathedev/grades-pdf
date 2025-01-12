# app.py
from flask import Flask, render_template, request, send_file
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.backends.backend_pdf import PdfPages
import io
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    # Get form data
    subject_code = request.form['subject_code']
    subject_name = request.form['subject_name']

    # Get grades data
    grades = ['AA', 'AB', 'BB', 'BC', 'CC', 'CD', 'DD', 'FF']
    theory = [int(request.form[f'theory_{grade}']) for grade in grades]
    practical = [int(request.form[f'practical_{grade}']) for grade in grades]

    # Generate PDF using the plotting code
    pdf_buffer = generate_grade_analysis(grades, theory, practical, subject_code, subject_name)

    # Send the PDF file
    pdf_buffer.seek(0)
    return send_file(
        pdf_buffer,
        download_name='Grade_Analysis.pdf',
        mimetype='application/pdf',
        as_attachment=True
    )

def generate_grade_analysis(grades, theory, practical, subject_code, subject_name):
    # Calculate percentages
    total_theory = sum(theory)
    total_practical = sum(practical)
    theory_percentage = [(count / total_theory) * 100 for count in theory]
    practical_percentage = [(count / total_practical) * 100 for count in practical]

    # Create DataFrame
    data = {
        "Grade": grades,
        "Theory Count": theory,
        "Practical Count": practical,
        "Theory %": [f"{perc:.1f}%" for perc in theory_percentage],
        "Practical %": [f"{perc:.1f}%" for perc in practical_percentage],
    }
    df = pd.DataFrame(data)

    # Create figure
    fig, ax = plt.subplots(2, 1, figsize=(9, 12))

    # Bar graph
    x = np.arange(len(grades))
    width = 0.35

    bars_theory = ax[0].bar(x - width/2, theory, width, label="Theory", color="blue")
    bars_practical = ax[0].bar(x + width/2, practical, width, label="Practical", color="orange")

    # Annotate bars
    for i, bar in enumerate(bars_theory):
        if bar.get_height() > 3:
            ax[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                      f"{theory[i]}\n({theory_percentage[i]:.1f}%)", ha='center', va='bottom', fontsize=9)
        else:
            ax[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f"{theory[i]}\n({theory_percentage[i]:.1f}%)", ha='center', va='bottom', fontsize=9)

    for i, bar in enumerate(bars_practical):
        if bar.get_height() > 3:
            ax[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f"{practical[i]}\n({practical_percentage[i]:.1f}%)", ha='left', va='bottom', fontsize=9)
        else:
            ax[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f"{practical[i]}\n({practical_percentage[i]:.1f}%)", ha='left', va='bottom', fontsize=9)

    # Adjust y-axis limit
    max_height = max(max(theory), max(practical))
    ax[0].set_ylim(0, max_height + 8)

    # Gaussian smoothing
    x_smooth = np.linspace(0, len(grades) - 1, 200)
    kde_theory = gaussian_kde(np.arange(len(grades)), weights=theory)
    kde_practical = gaussian_kde(np.arange(len(grades)), weights=practical)
    theory_smooth = kde_theory(x_smooth)
    practical_smooth = kde_practical(x_smooth)

    ax[0].plot(x_smooth, theory_smooth / max(theory_smooth) * max(theory),
               color='blue', linestyle='-', label='Theory (Smoothed)')
    ax[0].plot(x_smooth, practical_smooth / max(practical_smooth) * max(practical),
               color='orange', linestyle='--', label='Practical (Smoothed)')

    # Customize graph
    ax[0].set_title(f'{subject_code}: {subject_name} - Grade Analysis of Theory and Practical (A.Y. 2024-25 ODD)',
                    fontsize=12)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(grades)
    ax[0].set_xlabel("Grades", fontsize=12)
    ax[0].set_ylabel("Counts", fontsize=12)
    ax[0].legend()
    ax[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Table
    ax[1].axis("tight")
    ax[1].axis("off")
    table = ax[1].table(cellText=df.values, colLabels=df.columns,
                       cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Style table
    for col, cell in table.get_celld().items():
        if col[0] == 0:
            cell.set_text_props(weight='bold')

    for i, key in enumerate(table.get_celld().keys()):
        row, col = key
        if row == 0:
            table[row, col].set_height(0.09)
        else:
            table[row, col].set_height(0.07)

    # Save to buffer
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        pdf.savefig(fig)
    plt.close()

    return pdf_buffer

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8030, debug=True)
