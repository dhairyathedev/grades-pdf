# app.py
from flask import Flask, render_template, request, send_file
import matplotlib
matplotlib.use('Agg')  # Required for Lambda environment
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.backends.backend_pdf import PdfPages
import io
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    try:
        # Get form data
        subject_code = request.form.get('subject_code', '')
        subject_name = request.form.get('subject_name', '')

        # Get grades data
        grades = ['AA', 'AB', 'BB', 'BC', 'CC', 'CD', 'DD', 'FF']
        
        # Handle empty inputs by first converting to '0' string, then to int
        theory = [int(request.form.get(f'theory_{grade}', '0') or '0') for grade in grades]
        practical = [int(request.form.get(f'practical_{grade}', '0') or '0') for grade in grades]

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
    except Exception as e:
        print(f"Error: {str(e)}")
        return f"Error generating PDF: {str(e)}", 500

def generate_grade_analysis(grades, theory, practical, subject_code, subject_name):
    # Filter out grades with zero counts
    valid_indices = [i for i in range(len(grades)) if theory[i] > 0 or practical[i] > 0]
    filtered_grades = [grades[i] for i in valid_indices]
    filtered_theory = [theory[i] for i in valid_indices]
    filtered_practical = [practical[i] for i in valid_indices]

    # Calculate percentages using filtered data
    total_theory = sum(filtered_theory)
    total_practical = sum(filtered_practical)
    theory_percentage = [(count / total_theory) * 100 if total_theory > 0 else 0 for count in filtered_theory]
    practical_percentage = [(count / total_practical) * 100 if total_practical > 0 else 0 for count in filtered_practical]

    # Create DataFrame with filtered data
    data = {
        "Grade": filtered_grades,
        "Theory Count": filtered_theory,
        "Practical Count": filtered_practical,
        "Theory %": [f"{perc:.1f}%" for perc in theory_percentage],
        "Practical %": [f"{perc:.1f}%" for perc in practical_percentage],
    }
    df = pd.DataFrame(data)

    # Create figure
    fig, ax = plt.subplots(2, 1, figsize=(9, 12))

    # Bar graph
    x = np.arange(len(filtered_grades))
    width = 0.35

    bars_theory = ax[0].bar(x - width/2, filtered_theory, width, label="Theory", color="blue")
    bars_practical = ax[0].bar(x + width/2, filtered_practical, width, label="Practical", color="orange")

    # Annotate bars
    for i, bar in enumerate(bars_theory):
        if bar.get_height() > 3:
            ax[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                      f"{filtered_theory[i]}\n({theory_percentage[i]:.1f}%)", ha='center', va='bottom', fontsize=9)
        else:
            ax[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f"{filtered_theory[i]}\n({theory_percentage[i]:.1f}%)", ha='center', va='bottom', fontsize=9)

    for i, bar in enumerate(bars_practical):
        if bar.get_height() > 3:
            ax[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f"{filtered_practical[i]}\n({practical_percentage[i]:.1f}%)", ha='left', va='bottom', fontsize=9)
        else:
            ax[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f"{filtered_practical[i]}\n({practical_percentage[i]:.1f}%)", ha='left', va='bottom', fontsize=9)

    # Adjust y-axis limit
    max_height = max(max(filtered_theory), max(filtered_practical))
    ax[0].set_ylim(0, max_height + 8)

    # Gaussian smoothing with filtered data
    x_smooth = np.linspace(0, len(filtered_grades) - 1, 200)
    if len(filtered_grades) > 1:  # Only apply smoothing if we have more than one grade
        kde_theory = gaussian_kde(np.arange(len(filtered_grades)), weights=filtered_theory)
        kde_practical = gaussian_kde(np.arange(len(filtered_grades)), weights=filtered_practical)
        theory_smooth = kde_theory(x_smooth)
        practical_smooth = kde_practical(x_smooth)

        ax[0].plot(x_smooth, theory_smooth / max(theory_smooth) * max(filtered_theory),
                   color='blue', linestyle='-', label='Theory (Smoothed)')
        ax[0].plot(x_smooth, practical_smooth / max(practical_smooth) * max(filtered_practical),
                   color='orange', linestyle='--', label='Practical (Smoothed)')

    # Customize graph
    ax[0].set_title(f'{subject_code}: {subject_name} - Grade Analysis of Theory and Practical (A.Y. 2024-25 ODD)',
                    fontsize=12)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(filtered_grades)
    ax[0].set_xlabel("Grades", fontsize=12)
    ax[0].set_ylabel("Counts", fontsize=12)
    ax[0].legend()
    ax[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Create table data
    table_data = [
        ['Grade', 'Theory Count', 'Practical Count', 'Theory %', 'Practical %'],
    ]
    for i in range(len(filtered_grades)):
        table_data.append([
            filtered_grades[i],
            str(filtered_theory[i]),
            str(filtered_practical[i]),
            f"{theory_percentage[i]:.1f}%",
            f"{practical_percentage[i]:.1f}%"
        ])

    # Create table
    ax[1].axis('tight')
    ax[1].axis('off')
    table = ax[1].table(cellText=table_data, cellLoc='center', loc='center')
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for i in range(len(table_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_height(0.09)
            else:
                cell.set_height(0.07)

    # Save to buffer
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        pdf.savefig(fig)
    plt.close()

    return pdf_buffer

if __name__ == '__main__':
    app.run(debug=True)
