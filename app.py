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
        if dept == 'CSE' and sem == 3:
            max_th = dict(zip(grades, [23, 43, 48, 28, 21, 24, 55, 8]))
            avg_th = dict(zip(grades, [12, 28, 39, 25, 18, 15, 25, 8]))
            min_th = dict(zip(grades, [1, 10, 25, 21, 14, 7, 9, 0]))
            max_pr = dict(zip(grades, [67, 87, 40, 14, 1, 2, 0, 1]))
            avg_pr = dict(zip(grades, [52, 67, 37, 8, 1, 2, 0, 1]))
            min_pr = dict(zip(grades, [29, 53, 34, 5, 0, 1, 0, 1]))
        elif dept == 'CSE' and sem == 5:
            max_th = dict(zip(grades, [14, 25, 55, 55, 32, 19, 31, 2]))
            avg_th = dict(zip(grades, [7.5, 17, 47, 42.33, 26.33, 16.33, 18.66, 1.33]))
            min_th = dict(zip(grades, [1, 7, 41, 36, 22, 13, 11, 1]))
            max_pr = dict(zip(grades, [100, 96, 76, 32, 9, 38, 4, 1]))
            avg_pr = dict(zip(grades, [31.9, 48.44, 33.66, 9.8, 3, 14.66, 1.75, 1]))
            min_pr = dict(zip(grades, [7, 6, 9, 1, 1, 1, 1, 1]))
        elif dept == 'CSE' and sem == 7:
            max_th = dict(zip(grades, [16, 45, 63, 49, 20, 10, 11, 2]))
            avg_th = dict(zip(grades, [6.83, 27.33, 45.33, 29.5, 15.8, 6.33, 8.40, 2]))
            min_th = dict(zip(grades, [2, 18, 6, 5, 6, 3, 4, 2]))
            max_pr = dict(zip(grades, [113, 79, 68, 52, 12, 9, 0, 0]))
            avg_pr = dict(zip(grades, [41.65, 49.75,34.25, 14.85, 3.6, 3.5, 0, 0]))
            min_pr = dict(zip(grades, [29, 53, 34, 5, 0, 1, 0, 1]))
        elif dept == 'CE' and sem == 3:
            max_th = dict(zip(grades, [35,	26,	31,	31,	26,	24,	50,	5]))
            avg_th = dict(zip(grades, [17,	16,	25.75,	26.5,	20,	15,	37,	4.25]))
            min_th = dict(zip(grades, [3,	11,	19,	19,	12,	6,	25,	2]))
            max_pr = dict(zip(grades, [55,	83,	71,	35,	22,	0,	0,	0]))
            avg_pr = dict(zip(grades, [37.7,	51.5,	47.75,	19,	11.5,	0,	0,	0]))
            min_pr = dict(zip(grades, [17,	24,	35,	3,	1,	0,	0,	0]))
        elif dept == 'CE' and sem == 5:
            max_th = dict(zip(grades, [49,	44,	47,	30,	23,	12,	17,	3]))
            avg_th = dict(zip(grades, [13.16666667,	22,	22.5,	18.33333333,	12.5,	7,	10.66666667,	2.333333333]))
            min_th = dict(zip(grades, [1,	5,	6,	3,	3,	1,	1,	2]))
            max_pr = dict(zip(grades, [75,	85,	48,	97,	18,	2,	0,	0]))
            avg_pr = dict(zip(grades, [26.55555556,	33.55555556,	25.77777778,	23.57142857,	8.666666667,	2,	0,	0]))
            min_pr = dict(zip(grades, [3, 1, 6, 1, 3, 2, 0, 0]))
        elif dept == "CE" and sem == 7:
            max_th = dict(zip(grades, [10, 25, 55, 59, 28, 22, 27, 1]))
            avg_th = dict(zip(grades, [6, 15.33, 31, 31.66, 22.4, 12, 14.2, 1]))
            min_th = dict(zip(grades, [2, 2, 8, 2, 10, 2, 6, 1]))
            max_pr = dict(zip(grades, [70, 89, 94, 30, 10, 22, 0,0]))
            avg_pr = dict(zip(grades, [35.875, 40.375, 41.750, 14.285, 6, 7.500]))
            min_pr = dict(zip(grades, [2,4,5,1,4,1]))
        elif dept == "IT" and sem == 3:
            max_th = dict(zip(grades, [22, 19, 28, 20, 17, 16, 30, 2]))
            avg_th = dict(zip(grades, [6, 10, 13, 11, 9, 8, 12, 2]))
            min_th = dict(zip(grades, [5, 1, 8, 9, 8, 5, 2, 2]))
            max_pr = dict(zip(grades, [53, 35, 32, 16, 20, 17, 2, 0]))
            avg_pr = dict(zip(grades, [29, 20, 14, 5, 4, 3, 0, 0]))
            min_pr = dict(zip(grades, [16, 34, 25, 3, 1, 0, 0, 0]))
        elif dept == "IT" and sem == 5:
            max_th = dict(zip(grades, [36, 28, 35, 27, 15, 21, 12, 0]))
            avg_th = dict(zip(grades, [6, 15, 15, 12, 8, 8, 5, 0]))
            min_th = dict(zip(grades, [1, 4, 8, 10, 3, 2, 4, 0]))
            max_pr = dict(zip(grades, [86, 46, 33, 18, 28, 8, 1, 0]))
            avg_pr = dict(zip(grades, [37, 19, 16, 5, 4, 1, 1, 0]))
            min_pr = dict(zip(grades, [12, 5, 8, 2, 1, 1, 0, 0]))
        elif dept == "IT" and sem == 7:
            max_th = dict(zip(grades, [10, 34, 40, 36, 15, 25, 28, 5]))
            avg_th = dict(zip(grades, [2, 9, 19, 16, 11, 6, 6, 1]))
            min_th = dict(zip(grades, [1, 3, 6, 9, 7, 2, 3, 1]))
            max_pr = dict(zip(grades, [66, 48, 60, 40, 7, 3, 0, 0]))
            avg_pr = dict(zip(grades, [33, 32, 24, 8, 1, 1, 0, 0]))
            min_pr = dict(zip(grades, [4, 20, 5, 1, 1, 1, 0, 0]))
        else:
            # Default statistics if department and semester combination is not found
            max_th = dict(zip(grades, [50, 50, 50, 50, 50, 50, 50, 50]))
            avg_th = dict(zip(grades, [25, 25, 25, 25, 25, 25, 25, 25]))
            min_th = dict(zip(grades, [0, 0, 0, 0, 0, 0, 0, 0]))
            max_pr = dict(zip(grades, [50, 50, 50, 50, 50, 50, 50, 50]))
            avg_pr = dict(zip(grades, [25, 25, 25, 25, 25, 25, 25, 25]))
            min_pr = dict(zip(grades, [0, 0, 0, 0, 0, 0, 0, 0]))
        
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
        if len(filtered_grades) >= 2:  # Only add smoothing if we have 3 or more grades
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

        # Get the requested format
        output_format = request.form.get('format', 'pdf')

        # Save to memory buffer
        buf = io.BytesIO()
        
        if output_format == 'pdf':
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
        else:  # PNG format
            # Save as PNG with higher DPI for better quality
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Prepare buffer for download
            buf.seek(0)
            return send_file(
                buf,
                download_name=f'Grade_Analysis_{subject_code}.png',
                mimetype='image/png'
            )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=8030, debug=True)
