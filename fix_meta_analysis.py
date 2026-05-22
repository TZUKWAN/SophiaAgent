import re

with open('sophia/research/meta_analysis.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Make sure matplotlib is imported
if "import matplotlib" not in content and "import matplotlib.pyplot" not in content:
    content = "import matplotlib.pyplot as plt\nimport os\n" + content
elif "import os" not in content:
    content = "import os\n" + content

# Add forest_plot method
forest_plot_code = """
    def forest_plot(self, args: Dict[str, Any]) -> str:
        \"\"\"Generate a forest plot for meta-analysis results and store it as an image.
        
        Args:
            study_names (list): Names of studies
            effects (list): Effect sizes
            ci_low (list): CI lower bounds
            ci_high (list): CI upper bounds
            pooled_effect (float): Overall pooled effect
            pooled_ci_low (float): Overall CI lower
            pooled_ci_high (float): Overall CI upper
        \"\"\"
        try:
            study_names = args["study_names"]
            effects = args["effects"]
            ci_low = args["ci_low"]
            ci_high = args["ci_high"]
            pooled_effect = args["pooled_effect"]
            pooled_ci_low = args["pooled_ci_low"]
            pooled_ci_high = args["pooled_ci_high"]
            
            n_studies = len(study_names)
            
            fig, ax = plt.subplots(figsize=(10, max(4, n_studies * 0.5 + 2)))
            
            # Plot individual studies
            y_pos = list(range(n_studies, 0, -1))
            
            # Error bars
            xerr = [
                [effects[i] - ci_low[i] for i in range(n_studies)],
                [ci_high[i] - effects[i] for i in range(n_studies)]
            ]
            ax.errorbar(effects, y_pos, xerr=xerr, fmt='s', color='blue', 
                       ecolor='black', capsize=0, markersize=8, label='Individual Studies')
            
            # Plot pooled effect (diamond)
            ax.plot(pooled_effect, 0, 'D', color='red', markersize=10, label='Pooled Effect')
            
            # Add line for pooled effect CI
            ax.errorbar(pooled_effect, 0, 
                       xerr=[[pooled_effect - pooled_ci_low], [pooled_ci_high - pooled_effect]], 
                       fmt='none', ecolor='red', capsize=0, linewidth=2)
            
            # Vertical line at null effect (usually 0 for differences, 1 for ratios)
            # Assuming mean difference or similar here. Could be parameterized.
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add horizontal line separating studies from pooled effect
            ax.axhline(y=0.5, color='black', linestyle='-')
            
            # Set labels and ticks
            ax.set_yticks([0] + y_pos)
            ax.set_yticklabels(['Overall'] + study_names)
            ax.set_xlabel('Effect Size')
            ax.set_title('Forest Plot of Meta-Analysis')
            
            # Add grid for easier reading
            ax.grid(True, axis='x', linestyle=':', alpha=0.6)
            
            plt.tight_layout()
            
            # Save to temporary path and then store
            import tempfile
            fd, tmp_path = tempfile.mkstemp(suffix='.png')
            os.close(fd)
            
            fig.savefig(tmp_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            if self.store:
                with open(tmp_path, 'rb') as f:
                    img_data = f.read()
                
                result_id = self.store.store(
                    "meta_forest_plot",
                    img_data,
                    meta={
                        "type": "image",
                        "description": "Forest plot summarizing meta-analysis",
                        "studies": len(study_names)
                    }
                )
                
                output = {
                    "status": "success",
                    "result_id": result_id,
                    "studies": len(study_names),
                    "pooled_effect": pooled_effect,
                    "message": f"Forest plot generated and saved with result id: {result_id}"
                }
            else:
                output = {
                    "status": "warning",
                    "path": tmp_path,
                    "message": "Forest plot generated but no ResultStore attached. Image at temporary file path."
                }
                
            return json.dumps(output, ensure_ascii=False)
            
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {str(e)}")
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Failed to generate forest plot: {str(e)}"})
"""

# Insert forest_plot method before the end of the class
# Look for a decent place, e.g. after publication_bias
if "def publication_bias" in content:
    idx = content.find("def publication_bias")
    # find next method start or class end
    next_def = content.find("    def ", idx + 20)
    if next_def == -1:
        # insert at end
        content += "\n" + forest_plot_code
    else:
        content = content[:next_def] + forest_plot_code + "\n" + content[next_def:]
else:
    # Just append to the file
    content += "\n" + forest_plot_code

with open('sophia/research/meta_analysis.py', 'w', encoding='utf-8') as f:
    f.write(content)
