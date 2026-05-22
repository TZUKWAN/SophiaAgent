import json
import subprocess
import tempfile
import os

class REngine:
    """
    R Engine to execute complex causal inference models in R via subprocess.
    Avoids rpy2 to ensure better stability across different host environments.
    """
    
    def __init__(self):
        pass

    def run_rd_robust(self, data_csv_path: str, y_var: str, x_var: str, cutoff: float) -> dict:
        """
        Executes Regression Discontinuity (RD) design using the rdrobust R package.
        
        Args:
            data_csv_path (str): Path to the CSV data file.
            y_var (str): Dependent variable column name.
            x_var (str): Running variable column name.
            cutoff (float): Cutoff point for the RD design.
            
        Returns:
            dict: JSON parsed output containing RD estimates.
        """
        # Ensure path uses forward slashes for R
        safe_path = data_csv_path.replace("\\", "/")
        
        # R script that manually escapes to JSON to avoid relying on jsonlite
        r_script = f"""
        # Attempt to load rdrobust
        if (!suppressWarnings(require("rdrobust", character.only = TRUE))) {{
            cat('{{"error": "Rscript execution failed", "details": "rdrobust package not found. Please install it in R."}}')
            quit(status = 0)
        }}

        tryCatch({{
            data <- read.csv("{safe_path}")
            
            if (!"{y_var}" %in% names(data)) {{
                stop(sprintf("Column '%s' not found in dataset.", "{y_var}"))
            }}
            if (!"{x_var}" %in% names(data)) {{
                stop(sprintf("Column '%s' not found in dataset.", "{x_var}"))
            }}
            
            # Run rdrobust
            rd_out <- rdrobust(y = data[[ "{y_var}" ]], x = data[[ "{x_var}" ]], c = {cutoff})
            
            # Extract relevant stats
            coef <- rd_out$coef[1]
            se <- rd_out$se[1]
            z <- rd_out$z[1]
            pv <- rd_out$pv[1]
            ci_lower <- rd_out$ci[1, 1]
            ci_upper <- rd_out$ci[1, 2]
            n_left <- rd_out$N[1]
            n_right <- rd_out$N[2]
            
            # Output manual JSON 
            cat(sprintf('{{"coef": %f, "se": %f, "z": %f, "p_value": %e, "ci_lower": %f, "ci_upper": %f, "n_left": %d, "n_right": %d}}',
                        coef, se, z, pv, ci_lower, ci_upper, n_left, n_right))
                        
        }}, error = function(e) {{
            # Escape quotes in error message
            msg <- gsub('"', '\\\\"', e$message)
            cat(sprintf('{{"error": "Rscript execution failed", "details": "%s"}}', msg))
        }})
        """
        
        # Write the script to a temporary file
        fd, tmp_file = tempfile.mkstemp(suffix=".R")
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(r_script)
            
        try:
            # Run Rscript and capture output
            result = subprocess.run(
                ["Rscript", tmp_file],
                capture_output=True,
                text=True,
                check=False  # Handled by checking return code or parsing output
            )
            
            # Extract output
            output = result.stdout.strip()
            
            # In case Rscript fails before executing (e.g., syntax error, non-existent Rscript)
            if result.returncode != 0 and not output:
                return {
                    "error": "Rscript execution failed",
                    "details": result.stderr.strip() or "Unknown Rscript error"
                }

            # Parse the JSON response
            try:
                # Sometimes there might be warning logs in stdout before the JSON
                # We specifically look for the { bracket
                if "{" in output:
                    json_str = output[output.find("{"):output.rfind("}")+1]
                    return json.loads(json_str)
                else:
                    return {
                        "error": "Rscript execution failed",
                        "details": "Could not find JSON in Rscript output.",
                        "stdout": output,
                        "stderr": result.stderr.strip()
                    }
                    
            except json.JSONDecodeError as je:
                return {
                    "error": "Rscript execution failed",
                    "details": f"Failed to parse JSON output: {str(je)}",
                    "stdout": output,
                    "stderr": result.stderr.strip()
                }
                
        except FileNotFoundError:
            return {
                "error": "Rscript execution failed",
                "details": "Rscript command not found. Ensure R is installed and in your PATH."
            }
        except Exception as e:
            return {
                "error": "Rscript execution failed",
                "details": str(e)
            }
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
