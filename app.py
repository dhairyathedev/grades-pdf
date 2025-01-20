import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot

from flask import Flask, render_template, request, send_file
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import io
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        subject_code = request.form['subject_code']
        subject_name = request.form['subject_name']
        dept = request.form['dept']
        sem = int(request.form['sem'])
        
        # Get grade counts and filter out empty ones
        grades = ['AA', 'AB', 'BB', 'BC', 'CC', 'CD', 'DD', 'FF']
        theory = [int(request.form[f'theory_{grade}']) for grade in grades]
        practical = [int(request.form[f'practical_{grade}']) for grade in grades]
        
        # Filter out grades with zero counts for both theory and practical
        valid_indices = [i for i in range(len(grades)) 
                        if theory[i] > 0 or practical[i] > 0]
        
        if not valid_indices:
            return "Please enter at least one grade count", 400
            
        # Filter arrays to only include grades with non-zero counts
        filtered_grades = [grades[i] for i in valid_indices]
        filtered_theory = [theory[i] for i in valid_indices]
        filtered_practical = [practical[i] for i in valid_indices]

        # Get department statistics for valid grades only
        if dept == 'CE' and sem == 3:
            avg_th = dict(zip(grades, [17, 29, 35, 24, 17, 14, 25, 2]))
            max_th = dict(zip(grades, [45, 40, 55, 50, 48, 38, 42, 30]))
            min_th = dict(zip(grades, [5, 10, 8, 4, 6, 7, 9, 2]))
            avg_pr = dict(zip(grades, [22, 23, 10, 45, 22, 23, 10, 45]))
            max_pr = dict(zip(grades, [70, 75, 50, 60, 40, 30, 20, 10]))
            min_pr = dict(zip(grades, [10, 5, 3, 2, 1, 1, 0, 0]))
        elif dept == 'CE' and sem == 5:
            avg_th = dict(zip(grades, [19, 29, 35, 24, 17, 14, 25, 2]))
            max_th = dict(zip(grades, [110, 40, 55, 50, 48, 38, 42, 30]))
            min_th = dict(zip(grades, [5, 10, 8, 4, 6, 7, 9, 2]))
            avg_pr = dict(zip(grades, [22, 23, 10, 45, 22, 23, 10, 45]))
            max_pr = dict(zip(grades, [70, 75, 50, 60, 40, 30, 20, 10]))
            min_pr = dict(zip(grades, [10, 5, 3, 2, 1, 1, 0, 0]))

        # Filter statistics to match valid grades
        filtered_avg_th = [avg_th[filtered_grades[i]] for i in range(len(filtered_grades))]
        filtered_max_th = [max_th[filtered_grades[i]] for i in range(len(filtered_grades))]
        filtered_min_th = [min_th[filtered_grades[i]] for i in range(len(filtered_grades))]
        filtered_avg_pr = [avg_pr[filtered_grades[i]] for i in range(len(filtered_grades))]
        filtered_max_pr = [max_pr[filtered_grades[i]] for i in range(len(filtered_grades))]
        filtered_min_pr = [min_pr[filtered_grades[i]] for i in range(len(filtered_grades))]

        # Calculate percentages only for non-zero totals
        total_theory = sum(filtered_theory)
        total_practical = sum(filtered_practical)
        
        theory_percentage = [0] * len(filtered_theory)
        if total_theory > 0:
            theory_percentage = [(count / total_theory) * 100 for count in filtered_theory]
            
        practical_percentage = [0] * len(filtered_practical)
        if total_practical > 0:
            practical_percentage = [(count / total_practical) * 100 for count in filtered_practical]

        # Create DataFrame for table
        data = {
            "Grade": filtered_grades,
        }
        
        if total_theory > 0:
            data["Theory\nCount & (%)"] = [f"{count} ({perc:.1f}%)" for count, perc in zip(filtered_theory, theory_percentage)]
            data["Max\nTheory"] = filtered_max_th
            data["Avg\nTheory"] = filtered_avg_th
            data["Min\nTheory"] = filtered_min_th
            
        if total_practical > 0:
            data["Practical\nCount & (%)"] = [f"{count} ({perc:.1f}%)" for count, perc in zip(filtered_practical, practical_percentage)]
            data["Max\nPractical"] = filtered_max_pr
            data["Avg\nPractical"] = filtered_avg_pr
            data["Min\nPractical"] = filtered_min_pr

        df = pd.DataFrame(data)

        # Create figure
        fig, ax = plt.subplots(2, 1, figsize=(8.5, 12))
        plt.subplots_adjust(hspace=0.3)

        # Bar graph
        x = np.arange(len(filtered_grades))
        width = 0.35

        # Create bars only for non-zero totals
        bars_theory = None
        bars_practical = None
        
        if total_theory > 0:
            bars_theory = ax[0].bar(x - width/2, filtered_theory, width, label="Theory", color="blue")
            
        if total_practical > 0:
            bars_practical = ax[0].bar(x + width/2, filtered_practical, width, label="Practical", color="orange")

        # Annotate bars
        if bars_theory:
            for i, bar in enumerate(bars_theory):
                ax[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          f"{filtered_theory[i]}\n({theory_percentage[i]:.1f}%)", 
                          ha='center', va='bottom', fontsize=9)

        if bars_practical:
            for i, bar in enumerate(bars_practical):
                ax[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          f"{filtered_practical[i]}\n({practical_percentage[i]:.1f}%)", 
                          ha='left', va='bottom', fontsize=9)

        # Add markers only for non-zero totals
        for i in range(len(filtered_grades)):
            if total_theory > 0:
                theory_x = x[i] - width/2
                ax[0].plot([theory_x - 0.1, theory_x + 0.1], [filtered_avg_th[i], filtered_avg_th[i]], 
                          color="lightgreen", linewidth=2)
                center_x = (theory_x - 0.1 + theory_x + 0.1) / 2
                ax[0].plot(center_x, filtered_avg_th[i], color="lightgreen", marker='o')

                ax[0].plot([theory_x - 0.1, theory_x + 0.1], [filtered_max_th[i], filtered_max_th[i]], 
                          color="fuchsia", linewidth=2)
                ax[0].plot(center_x, filtered_max_th[i], color="fuchsia", marker='^')

                ax[0].plot([theory_x - 0.1, theory_x + 0.1], [filtered_min_th[i], filtered_min_th[i]], 
                          color="yellow", linewidth=2)
                ax[0].plot(center_x, filtered_min_th[i], color="yellow", marker='x')

            if total_practical > 0:
                practical_x = x[i] + width/2
                ax[0].plot([practical_x - 0.1, practical_x + 0.1], [filtered_avg_pr[i], filtered_avg_pr[i]], 
                          color="purple", linewidth=2)
                center_x = (practical_x - 0.1 + practical_x + 0.1) / 2
                ax[0].plot(center_x, filtered_avg_pr[i], color="purple", marker='o')

                ax[0].plot([practical_x - 0.1, practical_x + 0.1], [filtered_max_pr[i], filtered_max_pr[i]], 
                          color="maroon", linewidth=2)
                ax[0].plot(center_x, filtered_max_pr[i], color="maroon", marker='^')

                ax[0].plot([practical_x - 0.1, practical_x + 0.1], [filtered_min_pr[i], filtered_min_pr[i]], 
                          color="black", linewidth=2)
                ax[0].plot(center_x, filtered_min_pr[i], color="black", marker='x')

        # Adjust y-axis limit
        max_height = max(max(filtered_theory), max(filtered_practical), max(filtered_max_th), max(filtered_max_pr))
        ax[0].set_ylim(0, max_height + 10)

        # Add Gaussian smoothing only if we have enough data points
        if len(filtered_grades) > 2:  # Only add smoothing if we have 3 or more grades
            try:
                # Add small amount of noise to prevent singular matrices
                x_with_noise = x + np.random.normal(0, 0.01, len(x))
                
                if total_theory > 0:
                    kde_theory = gaussian_kde(x_with_noise, weights=filtered_theory, bw_method=0.5)
                    x_smooth = np.linspace(min(x), max(x), 200)
                    theory_smooth = kde_theory(x_smooth)
                    ax[0].plot(x_smooth, theory_smooth / max(theory_smooth) * max(filtered_theory),
                            color='blue', linestyle='-', label='Theory (Smoothed)')
                
                if total_practical > 0:
                    kde_practical = gaussian_kde(x_with_noise, weights=filtered_practical, bw_method=0.5)
                    practical_smooth = kde_practical(x_smooth)
                    ax[0].plot(x_smooth, practical_smooth / max(practical_smooth) * max(filtered_practical),
                            color='orange', linestyle='--', label='Practical (Smoothed)')
            except (np.linalg.LinAlgError, ValueError):
                # If KDE fails, skip the smoothing
                pass

        # Customize graph
        ax[0].set_title(f"{subject_code}: {subject_name}\nGrade Analysis of Theory and Practical (A.Y. 2024-25 ODD)",
                       fontsize=12)
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(filtered_grades)
        ax[0].set_xlabel("Grades", fontsize=12)
        ax[0].set_ylabel("Counts", fontsize=12)
        ax[0].grid(axis="y", linestyle="--", alpha=0.7)

        # Custom legend
        custom_legend_entry = [
            Line2D([0], [0], color="fuchsia", marker='^', linewidth=2, label="Th Max"),
            Line2D([0], [0], color="lightgreen", marker='o', linewidth=2, label="Th Avg"),
            Line2D([0], [0], color="yellow", marker='x', linewidth=2, label="Th Min"),
            Line2D([0], [0], color="maroon", marker='^', linewidth=2, label="Pr Max"),
            Line2D([0], [0], color="purple", marker='o', linewidth=2, label="Pr Avg"),
            Line2D([0], [0], color="black", marker='x', linewidth=2, label="Pr Min")
        ]

        handles, labels = ax[0].get_legend_handles_labels()
        handles.extend(custom_legend_entry)
        ax[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2),
                    fancybox=True, shadow=False, ncol=3, fontsize=10)

        # Table
        ax[1].axis("off")
        table = ax[1].table(cellText=df.values, colLabels=df.columns,
                           cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns))))

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

        # Add footnotes
        footnote1 = f"The Max, Avg, and Min values for Theory and Practical represent the maximum, average, and"
        footnote2 = f"minimum grade counts for a particular grade across all subjects in Semester {sem} of the {dept} department."
        fig.text(0.5, 0.115, footnote1, ha='center', va='center', fontsize=10, style='italic')
        fig.text(0.5, 0.1, footnote2, ha='center', va='center', fontsize=10, style='italic')

        # Save to memory buffer
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            pdf.savefig(fig)
        plt.close()

        # Prepare buffer for download
        buf.seek(0)
        return send_file(
            buf,
            download_name=f'Grade_Analysis_{subject_code}.pdf',
            mimetype='application/pdf'
        )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=8030, debug=True)
