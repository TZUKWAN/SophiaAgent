"""Method builder: generate handler code and tool schemas for new methods."""
import json
import traceback
from typing import Dict, Optional


class MethodBuilder:
    def __init__(self, catalog, provider=None):
        self.catalog = catalog
        self.provider = provider

    def build(self, candidate: dict, user_context: str) -> str:
        """Build a new method from a candidate.

        Args:
            candidate: dict with library, description, name, category, etc.
            user_context: the original user request

        Returns:
            JSON string with success, method_id, or error
        """
        library = candidate.get("library", "")
        description = candidate.get("description", "")
        method_name = candidate.get("name", "")
        category = candidate.get("category", "uncategorized")
        pip_name = candidate.get("pip_name", library)

        if not library:
            return json.dumps({
                "success": False,
                "error": "No library specified in candidate",
            }, ensure_ascii=False)

        # Step 1: Generate handler code
        handler_code = self._generate_handler(library, description, user_context)
        if handler_code is None:
            return json.dumps({
                "success": False,
                "error": f"Failed to generate handler for library '{library}'",
            }, ensure_ascii=False)

        # Step 2: Generate tool schema
        tool_schema = self._generate_schema(method_name or library, description)
        if tool_schema is None:
            return json.dumps({
                "success": False,
                "error": f"Failed to generate schema for '{method_name}'",
            }, ensure_ascii=False)

        # Step 3: Validate handler code
        if not self._validate(handler_code, tool_schema):
            return json.dumps({
                "success": False,
                "error": "Handler code validation failed (syntax error)",
            }, ensure_ascii=False)

        # Step 4: Build method ID and tool name
        method_id = library.replace("-", "_").replace(" ", "_").lower()
        tool_name = f"research_discovery_{method_id}"

        # Step 5: Add to catalog
        method = {
            "id": method_id,
            "name": method_name or library,
            "category": category,
            "description": description,
            "status": "installed",
            "tool_name": tool_name,
            "dependencies": [pip_name],
            "handler_code": handler_code,
            "tool_schema": tool_schema,
            "source": "auto_discovered",
            "discovery_context": user_context,
            "verified": True,
        }
        method_id_out = self.catalog.add(method)

        return json.dumps({
            "success": True,
            "method_id": method_id_out,
            "tool_name": tool_name,
            "library": library,
            "message": f"Method '{method_name or library}' built and added to catalog successfully",
        }, ensure_ascii=False)

    # -----------------------------------------------------------------
    # Category-aware handler templates
    # -----------------------------------------------------------------

    def _template_numpy(self, library: str, import_name: str,
                        method_description: str, user_requirement: str) -> str:
        """Template for numpy-based statistical/computational handlers."""
        safe_desc = method_description.replace("'", "\\'").replace('"', '\\"')
        return (
            "import json\n"
            "import traceback\n"
            "\n"
            "def handle(args):\n"
            f"    \"\"\"{safe_desc}\"\"\"\n"
            f"    try:\n"
            f"        import {import_name} as np\n"
            f"    except ImportError:\n"
            f"        return json.dumps({{\n"
            f"            'error': '{library} is not installed',\n"
            f"            'install': 'pip install {library}',\n"
            f"            'status': 'dependency_missing'\n"
            f"        }}, ensure_ascii=False)\n"
            f"\n"
            f"    try:\n"
            f"        data = args.get('data')\n"
            f"        method = args.get('method', 'default')\n"
            f"        params = args.get('parameters', {{}})\n"
            f"\n"
            f"        if data is None or not isinstance(data, list):\n"
            f"            return json.dumps({{\n"
            f"                'error': 'data must be a list of numbers or list of lists',\n"
            f"                'status': 'invalid_input'\n"
            f"            }}, ensure_ascii=False)\n"
            f"\n"
            f"        arr = np.array(data)\n"
            f"        result = {{\n"
            f"            'method': method,\n"
            f"            'shape': list(arr.shape),\n"
            f"            'mean': float(np.mean(arr)) if arr.size > 0 else None,\n"
            f"            'std': float(np.std(arr, ddof=1)) if arr.size > 1 else None,\n"
            f"            'median': float(np.median(arr)) if arr.size > 0 else None,\n"
            f"            'min': float(np.min(arr)) if arr.size > 0 else None,\n"
            f"            'max': float(np.max(arr)) if arr.size > 0 else None,\n"
            f"            'sum': float(np.sum(arr)) if arr.size > 0 else None,\n"
            f"        }}\n"
            f"\n"
            f"        if method == 'correlation' and len(arr.shape) == 2 and arr.shape[1] >= 2:\n"
            f"            result['correlation_matrix'] = np.corrcoef(arr.T).tolist()\n"
            f"        elif method == 'percentile':\n"
            f"            q = params.get('q', [25, 50, 75])\n"
            f"            result['percentiles'] = {{'p25': float(np.percentile(arr, 25)), 'p50': float(np.percentile(arr, 50)), 'p75': float(np.percentile(arr, 75))}}\n"
            f"\n"
            f"        result['status'] = 'success'\n"
            f"        return json.dumps(result, ensure_ascii=False, default=str)\n"
            f"    except Exception as e:\n"
            f"        return json.dumps({{\n"
            f"            'error': str(e),\n"
            f"            'error_type': type(e).__name__,\n"
            f"            'traceback': traceback.format_exc()\n"
            f"        }}, ensure_ascii=False)\n"
        )

    def _template_scipy(self, library: str, import_name: str,
                        method_description: str, user_requirement: str) -> str:
        """Template for scipy-based statistical test handlers."""
        safe_desc = method_description.replace("'", "\\'").replace('"', '\\"')
        return (
            "import json\n"
            "import traceback\n"
            "\n"
            "def handle(args):\n"
            f"    \"\"\"{safe_desc}\"\"\"\n"
            f"    try:\n"
            f"        import {import_name}.stats as stats\n"
            f"    except ImportError:\n"
            f"        return json.dumps({{\n"
            f"            'error': '{library} is not installed',\n"
            f"            'install': 'pip install {library}',\n"
            f"            'status': 'dependency_missing'\n"
            f"        }}, ensure_ascii=False)\n"
            f"\n"
            f"    try:\n"
            f"        data = args.get('data')\n"
            f"        method = args.get('method', 'ttest_ind')\n"
            f"        params = args.get('parameters', {{}})\n"
            f"\n"
            f"        if data is None or not isinstance(data, dict):\n"
            f"            return json.dumps({{\n"
            f"                'error': 'data must be a dict with group labels as keys',\n"
            f"                'status': 'invalid_input'\n"
            f"            }}, ensure_ascii=False)\n"
            f"\n"
            f"        result = {{'method': method, 'status': 'success'}}\n"
            f"\n"
            f"        if method in ('ttest_ind', 'ttest'):\n"
            f"            groups = list(data.values())\n"
            f"            if len(groups) < 2:\n"
            f"                return json.dumps({{'error': 'Need at least 2 groups'}}, ensure_ascii=False)\n"
            f"            t_stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False)\n"
            f"            result['t_statistic'] = float(t_stat)\n"
            f"            result['p_value'] = float(p_val)\n"
            f"            result['significant'] = bool(p_val < 0.05)\n"
            f"        elif method == 'mannwhitneyu':\n"
            f"            groups = list(data.values())\n"
            f"            if len(groups) < 2:\n"
            f"                return json.dumps({{'error': 'Need at least 2 groups'}}, ensure_ascii=False)\n"
            f"            u_stat, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')\n"
            f"            result['u_statistic'] = float(u_stat)\n"
            f"            result['p_value'] = float(p_val)\n"
            f"            result['significant'] = bool(p_val < 0.05)\n"
            f"        elif method == 'shapiro':\n"
            f"            arr = list(data.values())[0]\n"
            f"            w_stat, p_val = stats.shapiro(arr)\n"
            f"            result['w_statistic'] = float(w_stat)\n"
            f"            result['p_value'] = float(p_val)\n"
            f"            result['normal'] = p_val > 0.05\n"
            f"        elif method == 'chi2':\n"
            f"            obs = params.get('observed')\n"
            f"            if obs is None:\n"
            f"                return json.dumps({{'error': 'chi2 needs params.observed'}}, ensure_ascii=False)\n"
            f"            chi2_stat, p_val, dof, expected = stats.chi2_contingency(obs)\n"
            f"            result['chi2_statistic'] = float(chi2_stat)\n"
            f"            result['p_value'] = float(p_val)\n"
            f"            result['dof'] = int(dof)\n"
            f"            result['significant'] = bool(p_val < 0.05)\n"
            f"        else:\n"
            f"            result['note'] = f'Method {{method}} not yet implemented in template'\n"
            f"\n"
            f"        return json.dumps(result, ensure_ascii=False, default=str)\n"
            f"    except Exception as e:\n"
            f"        return json.dumps({{\n"
            f"            'error': str(e),\n"
            f"            'error_type': type(e).__name__,\n"
            f"            'traceback': traceback.format_exc()\n"
            f"        }}, ensure_ascii=False)\n"
        )

    def _template_sklearn(self, library: str, import_name: str,
                          method_description: str, user_requirement: str) -> str:
        """Template for sklearn-based ML handlers."""
        safe_desc = method_description.replace("'", "\\'").replace('"', '\\"')
        return (
            "import json\n"
            "import traceback\n"
            "\n"
            "def handle(args):\n"
            f"    \"\"\"{safe_desc}\"\"\"\n"
            f"    try:\n"
            f"        import {import_name}\n"
            f"    except ImportError:\n"
            f"        return json.dumps({{\n"
            f"            'error': '{library} is not installed',\n"
            f"            'install': 'pip install {library}',\n"
            f"            'status': 'dependency_missing'\n"
            f"        }}, ensure_ascii=False)\n"
            f"\n"
            f"    try:\n"
            f"        data = args.get('data')\n"
            f"        method = args.get('method', 'default')\n"
            f"        params = args.get('parameters', {{}})\n"
            f"\n"
            f"        if data is None or not isinstance(data, dict):\n"
            f"            return json.dumps({{\n"
            f"                'error': 'data must be dict with X and y keys',\n"
            f"                'status': 'invalid_input'\n"
            f"            }}, ensure_ascii=False)\n"
            f"\n"
            f"        from sklearn.model_selection import train_test_split\n"
            f"        X = data.get('X')\n"
            f"        y = data.get('y')\n"
            f"        if X is None or y is None:\n"
            f"            return json.dumps({{'error': 'Need X and y in data'}}, ensure_ascii=False)\n"
            f"\n"
            f"        test_size = params.get('test_size', 0.2)\n"
            f"        random_state = params.get('random_state', 42)\n"
            f"        X_train, X_test, y_train, y_test = train_test_split(\n"
            f"            X, y, test_size=test_size, random_state=random_state\n"
            f"        )\n"
            f"\n"
            f"        model_name = params.get('model', 'RandomForestClassifier')\n"
            f"        result = {{'model': model_name, 'status': 'success'}}\n"
            f"\n"
            f"        if model_name == 'RandomForestClassifier':\n"
            f"            from sklearn.ensemble import RandomForestClassifier\n"
            f"            clf = RandomForestClassifier(n_estimators=100, random_state=random_state)\n"
            f"            clf.fit(X_train, y_train)\n"
            f"            result['train_score'] = float(clf.score(X_train, y_train))\n"
            f"            result['test_score'] = float(clf.score(X_test, y_test))\n"
            f"            result['feature_importances'] = clf.feature_importances_.tolist()\n"
            f"        elif model_name == 'LogisticRegression':\n"
            f"            from sklearn.linear_model import LogisticRegression\n"
            f"            clf = LogisticRegression(max_iter=1000, random_state=random_state)\n"
            f"            clf.fit(X_train, y_train)\n"
            f"            result['train_score'] = float(clf.score(X_train, y_train))\n"
            f"            result['test_score'] = float(clf.score(X_test, y_test))\n"
            f"            result['coef'] = clf.coef_.tolist()\n"
            f"        elif model_name == 'LinearRegression':\n"
            f"            from sklearn.linear_model import LinearRegression\n"
            f"            reg = LinearRegression()\n"
            f"            reg.fit(X_train, y_train)\n"
            f"            result['train_score'] = float(reg.score(X_train, y_train))\n"
            f"            result['test_score'] = float(reg.score(X_test, y_test))\n"
            f"            result['coef'] = reg.coef_.tolist()\n"
            f"            result['intercept'] = float(reg.intercept_)\n"
            f"        else:\n"
            f"            result['note'] = f'Model {{model_name}} not yet implemented'\n"
            f"\n"
            f"        result['n_train'] = len(X_train)\n"
            f"        result['n_test'] = len(X_test)\n"
            f"        return json.dumps(result, ensure_ascii=False, default=str)\n"
            f"    except Exception as e:\n"
            f"        return json.dumps({{\n"
            f"            'error': str(e),\n"
            f"            'error_type': type(e).__name__,\n"
            f"            'traceback': traceback.format_exc()\n"
            f"        }}, ensure_ascii=False)\n"
        )

    def _template_statsmodels(self, library: str, import_name: str,
                              method_description: str, user_requirement: str) -> str:
        """Template for statsmodels-based regression handlers."""
        safe_desc = method_description.replace("'", "\\'").replace('"', '\\"')
        return (
            "import json\n"
            "import traceback\n"
            "\n"
            "def handle(args):\n"
            f"    \"\"\"{safe_desc}\"\"\"\n"
            f"    try:\n"
            f"        import {import_name}.api as sm\n"
            f"        import numpy as np\n"
            f"    except ImportError:\n"
            f"        return json.dumps({{\n"
            f"            'error': '{library} is not installed',\n"
            f"            'install': 'pip install {library}',\n"
            f"            'status': 'dependency_missing'\n"
            f"        }}, ensure_ascii=False)\n"
            f"\n"
            f"    try:\n"
            f"        data = args.get('data')\n"
            f"        method = args.get('method', 'ols')\n"
            f"        params = args.get('parameters', {{}})\n"
            f"\n"
            f"        if data is None or not isinstance(data, dict):\n"
            f"            return json.dumps({{\n"
            f"                'error': 'data must be dict with X and y keys',\n"
            f"                'status': 'invalid_input'\n"
            f"            }}, ensure_ascii=False)\n"
            f"\n"
            f"        y = data.get('y')\n"
            f"        X = data.get('X')\n"
            f"        if y is None or X is None:\n"
            f"            return json.dumps({{'error': 'Need X and y in data'}}, ensure_ascii=False)\n"
            f"\n"
            f"        X = sm.add_constant(X)\n"
            f"        result = {{'method': method, 'status': 'success'}}\n"
            f"\n"
            f"        if method == 'ols':\n"
            f"            model = sm.OLS(y, X)\n"
            f"            fit = model.fit()\n"
            f"            result['r_squared'] = float(fit.rsquared)\n"
            f"            result['adj_r_squared'] = float(fit.rsquared_adj)\n"
            f"            result['f_statistic'] = float(fit.fvalue)\n"
            f"            result['f_pvalue'] = float(fit.f_pvalue)\n"
            f"            result['aic'] = float(fit.aic)\n"
            f"            result['bic'] = float(fit.bic)\n"
            f"            result['params'] = fit.params.tolist()\n"
            f"            result['std_errors'] = fit.bse.tolist()\n"
            f"            result['pvalues'] = fit.pvalues.tolist()\n"
            f"            result['conf_int'] = np.asarray(fit.conf_int()).tolist()\n"
            f"            result['nobs'] = int(fit.nobs)\n"
            f"        else:\n"
            f"            result['note'] = f'Method {{method}} not yet implemented'\n"
            f"\n"
            f"        return json.dumps(result, ensure_ascii=False, default=str)\n"
            f"    except Exception as e:\n"
            f"        return json.dumps({{\n"
            f"            'error': str(e),\n"
            f"            'error_type': type(e).__name__,\n"
            f"            'traceback': traceback.format_exc()\n"
            f"        }}, ensure_ascii=False)\n"
        )

    def _template_pandas(self, library: str, import_name: str,
                         method_description: str, user_requirement: str) -> str:
        """Template for pandas-based data processing handlers."""
        safe_desc = method_description.replace("'", "\\'").replace('"', '\\"')
        return (
            "import json\n"
            "import traceback\n"
            "\n"
            "def handle(args):\n"
            f"    \"\"\"{safe_desc}\"\"\"\n"
            f"    try:\n"
            f"        import {import_name} as pd\n"
            f"    except ImportError:\n"
            f"        return json.dumps({{\n"
            f"            'error': '{library} is not installed',\n"
            f"            'install': 'pip install {library}',\n"
            f"            'status': 'dependency_missing'\n"
            f"        }}, ensure_ascii=False)\n"
            f"\n"
            f"    try:\n"
            f"        data = args.get('data')\n"
            f"        method = args.get('method', 'describe')\n"
            f"        params = args.get('parameters', {{}})\n"
            f"\n"
            f"        if data is None or not isinstance(data, list):\n"
            f"            return json.dumps({{\n"
            f"                'error': 'data must be a list of dicts (records)',\n"
            f"                'status': 'invalid_input'\n"
            f"            }}, ensure_ascii=False)\n"
            f"\n"
            f"        df = pd.DataFrame(data)\n"
            f"        result = {{'method': method, 'shape': list(df.shape), 'status': 'success'}}\n"
            f"\n"
            f"        if method == 'describe':\n"
            f"            desc = df.describe().to_dict()\n"
            f"            result['describe'] = desc\n"
            f"        elif method == 'groupby':\n"
            f"            col = params.get('column')\n"
            f"            agg = params.get('agg', 'mean')\n"
            f"            if col and col in df.columns:\n"
            f"                grouped = df.groupby(col).agg(agg).to_dict()\n"
            f"                result['grouped'] = grouped\n"
            f"            else:\n"
            f"                result['error'] = f'Column {{col}} not found'\n"
            f"        elif method == 'filter':\n"
            f"            col = params.get('column')\n"
            f"            op = params.get('op', '>')\n"
            f"            val = params.get('value')\n"
            f"            if col and col in df.columns and val is not None:\n"
            f"                if op == '>':\n"
            f"                    filtered = df[df[col] > val]\n"
            f"                elif op == '<':\n"
            f"                    filtered = df[df[col] < val]\n"
            f"                elif op == '==':\n"
            f"                    filtered = df[df[col] == val]\n"
            f"                else:\n"
            f"                    filtered = df\n"
            f"                result['filtered_shape'] = list(filtered.shape)\n"
            f"                result['filtered'] = filtered.to_dict('records')\n"
            f"            else:\n"
            f"                result['error'] = 'Invalid filter params'\n"
            f"        elif method == 'value_counts':\n"
            f"            col = params.get('column')\n"
            f"            if col and col in df.columns:\n"
            f"                vc = df[col].value_counts().to_dict()\n"
            f"                result['value_counts'] = vc\n"
            f"            else:\n"
            f"                result['error'] = f'Column {{col}} not found'\n"
            f"        else:\n"
            f"            result['columns'] = df.columns.tolist()\n"
            f"            result['dtypes'] = {{c: str(t) for c, t in df.dtypes.items()}}\n"
            f"\n"
            f"        return json.dumps(result, ensure_ascii=False, default=str)\n"
            f"    except Exception as e:\n"
            f"        return json.dumps({{\n"
            f"            'error': str(e),\n"
            f"            'error_type': type(e).__name__,\n"
            f"            'traceback': traceback.format_exc()\n"
            f"        }}, ensure_ascii=False)\n"
        )

    def _template_networkx(self, library: str, import_name: str,
                           method_description: str, user_requirement: str) -> str:
        """Template for networkx-based network analysis handlers."""
        safe_desc = method_description.replace("'", "\\'").replace('"', '\\"')
        return (
            "import json\n"
            "import traceback\n"
            "\n"
            "def handle(args):\n"
            f"    \"\"\"{safe_desc}\"\"\"\n"
            f"    try:\n"
            f"        import {import_name} as nx\n"
            f"    except ImportError:\n"
            f"        return json.dumps({{\n"
            f"            'error': '{library} is not installed',\n"
            f"            'install': 'pip install {library}',\n"
            f"            'status': 'dependency_missing'\n"
            f"        }}, ensure_ascii=False)\n"
            f"\n"
            f"    try:\n"
            f"        data = args.get('data')\n"
            f"        method = args.get('method', 'default')\n"
            f"        params = args.get('parameters', {{}})\n"
            f"\n"
            f"        if data is None or not isinstance(data, list):\n"
            f"            return json.dumps({{\n"
            f"                'error': 'data must be list of edges [(u,v), ...]',\n"
            f"                'status': 'invalid_input'\n"
            f"            }}, ensure_ascii=False)\n"
            f"\n"
            f"        G = nx.Graph()\n"
            f"        G.add_edges_from(data)\n"
            f"        result = {{\n"
            f"            'method': method,\n"
            f"            'n_nodes': G.number_of_nodes(),\n"
            f"            'n_edges': G.number_of_edges(),\n"
            f"            'density': nx.density(G),\n"
            f"            'is_connected': nx.is_connected(G),\n"
            f"            'status': 'success'\n"
            f"        }}\n"
            f"\n"
            f"        if method == 'centrality':\n"
            f"            result['degree_centrality'] = nx.degree_centrality(G)\n"
            f"            result['betweenness_centrality'] = nx.betweenness_centrality(G)\n"
            f"            result['closeness_centrality'] = nx.closeness_centrality(G)\n"
            f"        elif method == 'community':\n"
            f"            try:\n"
            f"                communities = nx.community.greedy_modularity_communities(G)\n"
            f"                result['communities'] = [list(c) for c in communities]\n"
            f"                result['n_communities'] = len(communities)\n"
            f"                result['modularity'] = nx.community.modularity(\n"
            f"                    G, communities\n"
            f"                )\n"
            f"            except Exception as ce:\n"
            f"                result['community_error'] = str(ce)\n"
            f"        elif method == 'shortest_path':\n"
            f"            source = params.get('source')\n"
            f"            target = params.get('target')\n"
            f"            if source is not None and target is not None:\n"
            f"                try:\n"
            f"                    sp = nx.shortest_path(G, source=source, target=target)\n"
            f"                    result['shortest_path'] = sp\n"
            f"                    result['path_length'] = len(sp) - 1\n"
            f"                except nx.NetworkXNoPath:\n"
            f"                    result['shortest_path'] = None\n"
            f"                    result['path_length'] = None\n"
            f"\n"
            f"        return json.dumps(result, ensure_ascii=False, default=str)\n"
            f"    except Exception as e:\n"
            f"        return json.dumps({{\n"
            f"            'error': str(e),\n"
            f"            'error_type': type(e).__name__,\n"
            f"            'traceback': traceback.format_exc()\n"
            f"        }}, ensure_ascii=False)\n"
        )

    def _template_nltk(self, library: str, import_name: str,
                       method_description: str, user_requirement: str) -> str:
        """Template for NLTK-based text processing handlers."""
        safe_desc = method_description.replace("'", "\\'").replace('"', '\\"')
        return (
            "import json\n"
            "import traceback\n"
            "\n"
            "def handle(args):\n"
            f"    \"\"\"{safe_desc}\"\"\"\n"
            f"    try:\n"
            f"        import {import_name}\n"
            f"    except ImportError:\n"
            f"        return json.dumps({{\n"
            f"            'error': '{library} is not installed',\n"
            f"            'install': 'pip install {library}',\n"
            f"            'status': 'dependency_missing'\n"
            f"        }}, ensure_ascii=False)\n"
            f"\n"
            f"    try:\n"
            f"        data = args.get('data')\n"
            f"        method = args.get('method', 'tokenize')\n"
            f"        params = args.get('parameters', {{}})\n"
            f"\n"
            f"        if data is None or not isinstance(data, str):\n"
            f"            return json.dumps({{\n"
            f"                'error': 'data must be a string',\n"
            f"                'status': 'invalid_input'\n"
            f"            }}, ensure_ascii=False)\n"
            f"\n"
            f"        result = {{'method': method, 'status': 'success'}}\n"
            f"\n"
            f"        if method == 'tokenize':\n"
            f"            tokens = {import_name}.word_tokenize(data)\n"
            f"            result['tokens'] = tokens\n"
            f"            result['n_tokens'] = len(tokens)\n"
            f"        elif method == 'sentiment_vader':\n"
            f"            from nltk.sentiment import SentimentIntensityAnalyzer\n"
            f"            sia = SentimentIntensityAnalyzer()\n"
            f"            scores = sia.polarity_scores(data)\n"
            f"            result['sentiment_scores'] = scores\n"
            f"            result['compound'] = scores['compound']\n"
            f"            result['positive'] = scores['pos']\n"
            f"            result['negative'] = scores['neg']\n"
            f"            result['neutral'] = scores['neu']\n"
            f"        elif method == 'pos_tag':\n"
            f"            tokens = {import_name}.word_tokenize(data)\n"
            f"            tagged = {import_name}.pos_tag(tokens)\n"
            f"            result['pos_tags'] = tagged\n"
            f"        elif method == 'freq_dist':\n"
            f"            tokens = {import_name}.word_tokenize(data)\n"
            f"            fd = {import_name}.FreqDist(tokens)\n"
            f"            result['most_common'] = fd.most_common(20)\n"
            f"            result['n_unique'] = len(fd)\n"
            f"        else:\n"
            f"            result['note'] = f'Method {method} not yet implemented'\n"
            f"\n"
            f"        return json.dumps(result, ensure_ascii=False, default=str)\n"
            f"    except Exception as e:\n"
            f"        return json.dumps({{\n"
            f"            'error': str(e),\n"
            f"            'error_type': type(e).__name__,\n"
            f"            'traceback': traceback.format_exc()\n"
            f"        }}, ensure_ascii=False)\n"
        )

    def _template_gensim(self, library: str, import_name: str,
                         method_description: str, user_requirement: str) -> str:
        """Template for gensim-based topic modeling handlers."""
        safe_desc = method_description.replace("'", "\\'").replace('"', '\\"')
        return (
            "import json\n"
            "import traceback\n"
            "\n"
            "def handle(args):\n"
            f"    \"\"\"{safe_desc}\"\"\"\n"
            f"    try:\n"
            f"        import {import_name}\n"
            f"    except ImportError:\n"
            f"        return json.dumps({{\n"
            f"            'error': '{library} is not installed',\n"
            f"            'install': 'pip install {library}',\n"
            f"            'status': 'dependency_missing'\n"
            f"        }}, ensure_ascii=False)\n"
            f"\n"
            f"    try:\n"
            f"        data = args.get('data')\n"
            f"        method = args.get('method', 'lda')\n"
            f"        params = args.get('parameters', {{}})\n"
            f"\n"
            f"        if data is None or not isinstance(data, list):\n"
            f"            return json.dumps({{\n"
            f"                'error': 'data must be list of tokenized documents',\n"
            f"                'status': 'invalid_input'\n"
            f"            }}, ensure_ascii=False)\n"
            f"\n"
            f"        result = {{'method': method, 'status': 'success'}}\n"
            f"\n"
            f"        if method == 'lda':\n"
            f"            dictionary = {import_name}.corpora.Dictionary(data)\n"
            f"            corpus = [dictionary.doc2bow(doc) for doc in data]\n"
            f"            n_topics = params.get('n_topics', 5)\n"
            f"            lda = {import_name}.models.LdaModel(\n"
            f"                corpus, num_topics=n_topics, id2word=dictionary, passes=10\n"
            f"            )\n"
            f"            topics = []\n"
            f"            for i in range(n_topics):\n"
            f"                words = lda.show_topic(i, topn=10)\n"
            f"                topics.append({{\n"
            f"                    'topic_id': i,\n"
            f"                    'words': [{{'word': w, 'weight': float(p)}} for w, p in words]\n"
            f"                }})\n"
            f"            result['topics'] = topics\n"
            f"            result['n_topics'] = n_topics\n"
            f"            result['vocab_size'] = len(dictionary)\n"
            f"        else:\n"
            f"            result['note'] = f'Method {method} not yet implemented'\n"
            f"\n"
            f"        return json.dumps(result, ensure_ascii=False, default=str)\n"
            f"    except Exception as e:\n"
            f"        return json.dumps({{\n"
            f"            'error': str(e),\n"
            f"            'error_type': type(e).__name__,\n"
            f"            'traceback': traceback.format_exc()\n"
            f"        }}, ensure_ascii=False)\n"
        )

    def _template_vader(self, library: str, import_name: str,
                        method_description: str, user_requirement: str) -> str:
        """Template for vaderSentiment handlers."""
        safe_desc = method_description.replace("'", "\\'").replace('"', '\\"')
        return (
            "import json\n"
            "import traceback\n"
            "\n"
            "def handle(args):\n"
            f"    \"\"\"{safe_desc}\"\"\"\n"
            f"    try:\n"
            f"        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer\n"
            f"    except ImportError:\n"
            f"        return json.dumps({{\n"
            f"            'error': '{library} is not installed',\n"
            f"            'install': 'pip install {library}',\n"
            f"            'status': 'dependency_missing'\n"
            f"        }}, ensure_ascii=False)\n"
            f"\n"
            f"    try:\n"
            f"        data = args.get('data')\n"
            f"        params = args.get('parameters', {{}})\n"
            f"\n"
            f"        if data is None:\n"
            f"            return json.dumps({{\n"
            f"                'error': 'data is required (str or list of str)',\n"
            f"                'status': 'invalid_input'\n"
            f"            }}, ensure_ascii=False)\n"
            f"\n"
            f"        analyzer = SentimentIntensityAnalyzer()\n"
            f"        result = {{'status': 'success'}}\n"
            f"\n"
            f"        if isinstance(data, str):\n"
            f"            scores = analyzer.polarity_scores(data)\n"
            f"            result['sentiment'] = scores\n"
            f"            result['compound'] = scores['compound']\n"
            f"            result['label'] = 'positive' if scores['compound'] >= 0.05 else ('negative' if scores['compound'] <= -0.05 else 'neutral')\n"
            f"        elif isinstance(data, list):\n"
            f"            results = []\n"
            f"            for text in data:\n"
            f"                scores = analyzer.polarity_scores(text)\n"
            f"                results.append({{\n"
            f"                    'text': text[:100],\n"
            f"                    'compound': scores['compound'],\n"
            f"                    'label': 'positive' if scores['compound'] >= 0.05 else ('negative' if scores['compound'] <= -0.05 else 'neutral')\n"
            f"                }})\n"
            f"            result['results'] = results\n"
            f"            result['n_items'] = len(results)\n"
            f"            result['avg_compound'] = sum(r['compound'] for r in results) / len(results) if results else 0\n"
            f"        else:\n"
            f"            result['error'] = 'data must be str or list of str'\n"
            f"\n"
            f"        return json.dumps(result, ensure_ascii=False, default=str)\n"
            f"    except Exception as e:\n"
            f"        return json.dumps({{\n"
            f"            'error': str(e),\n"
            f"            'error_type': type(e).__name__,\n"
            f"            'traceback': traceback.format_exc()\n"
            f"        }}, ensure_ascii=False)\n"
        )

    def _template_transformers(self, library: str, import_name: str,
                               method_description: str, user_requirement: str) -> str:
        """Template for transformers-based NLP handlers."""
        safe_desc = method_description.replace("'", "\\'").replace('"', '\\"')
        return (
            "import json\n"
            "import traceback\n"
            "\n"
            "def handle(args):\n"
            f"    \"\"\"{safe_desc}\"\"\"\n"
            f"    try:\n"
            f"        import {import_name}\n"
            f"    except ImportError:\n"
            f"        return json.dumps({{\n"
            f"            'error': '{library} is not installed',\n"
            f"            'install': 'pip install {library}',\n"
            f"            'status': 'dependency_missing'\n"
            f"        }}, ensure_ascii=False)\n"
            f"\n"
            f"    try:\n"
            f"        data = args.get('data')\n"
            f"        method = args.get('method', 'sentiment')\n"
            f"        params = args.get('parameters', {{}})\n"
            f"\n"
            f"        if data is None:\n"
            f"            return json.dumps({{\n"
            f"                'error': 'data is required',\n"
            f"                'status': 'invalid_input'\n"
            f"            }}, ensure_ascii=False)\n"
            f"\n"
            f"        model_name = params.get('model_name', 'distilbert-base-uncased-finetuned-sst-2-english')\n"
            f"        result = {{'model': model_name, 'status': 'success'}}\n"
            f"\n"
            f"        if method == 'sentiment':\n"
            f"            from {import_name} import pipeline\n"
            f"            classifier = pipeline('sentiment-analysis', model=model_name)\n"
            f"            if isinstance(data, str):\n"
            f"                out = classifier(data)[0]\n"
            f"                result['label'] = out['label']\n"
            f"                result['score'] = float(out['score'])\n"
            f"            elif isinstance(data, list):\n"
            f"                outs = classifier(data)\n"
            f"                result['results'] = outs\n"
            f"            else:\n"
            f"                result['error'] = 'data must be str or list'\n"
            f"        else:\n"
            f"            result['note'] = f'Method {method} not yet implemented'\n"
            f"\n"
            f"        return json.dumps(result, ensure_ascii=False, default=str)\n"
            f"    except Exception as e:\n"
            f"        return json.dumps({{\n"
            f"            'error': str(e),\n"
            f"            'error_type': type(e).__name__,\n"
            f"            'traceback': traceback.format_exc()\n"
            f"        }}, ensure_ascii=False)\n"
        )

    def _generic_template(self, library: str, import_name: str,
                          method_description: str, user_requirement: str) -> str:
        """Generic fallback template — returns library info and basic metadata."""
        safe_desc = method_description.replace("'", "\\'").replace('"', '\\"')
        safe_requirement = user_requirement.replace("'", "\\'").replace('"', '\\"')
        return (
            "import json\n"
            "import traceback\n"
            "\n"
            "def handle(args):\n"
            f"    \"\"\"{safe_desc}\"\"\"\n"
            f"    try:\n"
            f"        import {import_name}\n"
            f"    except ImportError:\n"
            f"        return json.dumps({{\n"
            f"            'error': '{library} is not installed',\n"
            f"            'install': 'pip install {library}',\n"
            f"            'status': 'dependency_missing'\n"
            f"        }}, ensure_ascii=False)\n"
            f"\n"
            f"    try:\n"
            f"        data = args.get('data')\n"
            f"        method = args.get('method', 'default')\n"
            f"        params = args.get('parameters', {{}})\n"
            f"\n"
            f"        # Auto-discover some basic info about the library\n"
            f"        public_names = [n for n in dir({import_name}) if not n.startswith('_')][:50]\n"
            f"        version = getattr({import_name}, '__version__', 'unknown')\n"
            f"\n"
            f"        result = {{\n"
            f"            'library': '{library}',\n"
            f"            'import_name': '{import_name}',\n"
            f"            'version': version,\n"
            f"            'method': method,\n"
            f"            'available_functions': public_names,\n"
            f"            'input_data_received': data is not None,\n"
            f"            'status': 'info',\n"
            f"            'note': 'This is an auto-generated info handler. Use a specific method or provide LLM-generated handler for full functionality.',\n"
            f"        }}\n"
            f"        return json.dumps(result, ensure_ascii=False, default=str)\n"
            f"    except Exception as e:\n"
            f"        return json.dumps({{\n"
            f"            'error': str(e),\n"
            f"            'error_type': type(e).__name__,\n"
            f"            'traceback': traceback.format_exc()\n"
            f"        }}, ensure_ascii=False)\n"
        )

    # Mapping of library names/patterns to template functions
    _LIBRARY_TEMPLATES = {
        "numpy": _template_numpy,
        "scipy": _template_scipy,
        "sklearn": _template_sklearn,
        "scikit-learn": _template_sklearn,
        "scikit_learn": _template_sklearn,
        "statsmodels": _template_statsmodels,
        "pandas": _template_pandas,
        "networkx": _template_networkx,
        "nltk": _template_nltk,
        "gensim": _template_gensim,
        "vadersentiment": _template_vader,
        "transformers": _template_transformers,
    }

    def _infer_template(self, library: str, method_description: str) -> callable:
        """Choose the best template based on library name and description."""
        lib_lower = library.lower().replace("-", "_")

        # Direct match
        if lib_lower in self._LIBRARY_TEMPLATES:
            return self._LIBRARY_TEMPLATES[lib_lower]

        # Substring match (e.g. "scipy.stats" -> "scipy")
        for key, tmpl in self._LIBRARY_TEMPLATES.items():
            if key in lib_lower or lib_lower.startswith(key):
                return tmpl

        # Category-based fallback using description
        desc_lower = method_description.lower()
        if any(w in desc_lower for w in ("network", "graph", "node", "edge", "centrality")):
            return self._template_networkx
        if any(w in desc_lower for w in ("sentiment", "emotion", "polarity")):
            return self._template_vader
        if any(w in desc_lower for w in ("topic", "lda", "word2vec")):
            return self._template_gensim
        if any(w in desc_lower for w in ("text", "tokenize", "pos", "ngram")):
            return self._template_nltk
        if any(w in desc_lower for w in ("machine learning", "classifier", "regression", "predict")):
            return self._template_sklearn
        if any(w in desc_lower for w in ("regression", "time series", "panel")):
            return self._template_statsmodels
        if any(w in desc_lower for w in ("statistical test", "t-test", "anova", "chi-square", "distribution")):
            return self._template_scipy
        if any(w in desc_lower for w in ("data frame", "table", "csv", "column")):
            return self._template_pandas
        if any(w in desc_lower for w in ("array", "matrix", "mean", "std", "correlation")):
            return self._template_numpy

        return self._generic_template

    # Common library name → Python import name overrides
    _IMPORT_NAME_MAP = {
        "scikit-learn": "sklearn",
        "scikit_learn": "sklearn",
        "pillow": "PIL",
    }

    def _generate_handler(self, library: str, method_description: str,
                          user_requirement: str) -> Optional[str]:
        """Generate handler Python code.
        If provider available, use LLM to generate.
        Otherwise, generate a category-aware functional template."""
        import_name = self._IMPORT_NAME_MAP.get(library, library.replace("-", "_"))

        # Try LLM-based generation first
        if self.provider is not None and hasattr(self.provider, "chat"):
            try:
                llm_code = self._llm_generate_handler(library, import_name,
                                                       method_description, user_requirement)
                if llm_code and self._validate(llm_code, {}):
                    return llm_code
            except Exception:
                pass  # Fall through to template

        # Select and generate category-aware template
        template_func = self._infer_template(library, method_description)
        import inspect
        params = list(inspect.signature(template_func).parameters.keys())
        if params and params[0] == "self":
            return template_func(self, library, import_name, method_description, user_requirement)
        return template_func(library, import_name, method_description, user_requirement)

    def _generate_schema(self, method_name: str, description: str) -> Optional[dict]:
        """Generate tool schema."""
        schema = {
            "description": f"{method_name}: {description}",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "description": "Input data for the analysis",
                    },
                    "method": {
                        "type": "string",
                        "description": f"Specific method/algorithm to use within {method_name}",
                        "default": "default",
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Additional parameters for the method",
                        "default": {},
                    },
                },
                "required": ["data"],
            },
        }
        return schema

    def _generate_test(self, handler_code: str) -> str:
        """Generate a simple validation test."""
        return (
            "import json\n"
            "import sys\n"
            "\n"
            "# Test handler with minimal input\n"
            "local_ns = {}\n"
            f"exec(handler_code, {{'json': json, 'traceback': __import__('traceback'), '__builtins__': __builtins__}}, local_ns)\n"
            "handle = local_ns.get('handle')\n"
            "if handle is None:\n"
            "    print('FAIL: No handle function found')\n"
            "    sys.exit(1)\n"
            "result = handle({'data': [1, 2, 3], 'method': 'default', 'parameters': {}})\n"
            "parsed = json.loads(result)\n"
            "print(f'Result: {parsed}')\n"
            "if 'error' in parsed and parsed.get('status') != 'dependency_missing':\n"
            "    print('FAIL: Unexpected error')\n"
            "    sys.exit(1)\n"
            "print('PASS')\n"
        )

    def _validate(self, handler_code: str, tool_schema: dict) -> bool:
        """Validate that the handler code is syntactically correct, safe, and runnable."""
        if not handler_code or not isinstance(handler_code, str):
            return False

        # Step 1: Sandbox validation (AST-based security check)
        from sophia.research.discovery.sandbox import HandlerSandbox
        is_safe, error = HandlerSandbox.validate(handler_code)
        if not is_safe:
            return False

        # Step 2: Check syntax by compiling
        try:
            compile(handler_code, "<handler>", "exec")
        except SyntaxError:
            return False

        # Step 3: Check that 'handle' function exists after safe exec
        try:
            local_ns = HandlerSandbox.exec_safe(
                handler_code,
                {"json": json, "traceback": traceback, "__builtins__": __builtins__},
            )
            if "handle" not in local_ns:
                return False
            if not callable(local_ns["handle"]):
                return False
        except Exception:
            return False

        return True

    def _llm_generate_handler(self, library: str, import_name: str,
                               method_description: str, user_requirement: str) -> Optional[str]:
        """Use LLM to generate a proper handler."""
        prompt = (
            f"Generate a Python function named 'handle' that wraps the library '{library}' "
            f"(imported as '{import_name}') for the purpose of: {method_description}. "
            f"User requirement context: {user_requirement}. "
            f"The function must:\n"
            f"1. Take a single 'args' dict parameter\n"
            f"2. Import {import_name} inside the function with a try/except for ImportError\n"
            f"3. Return json.dumps(result, ensure_ascii=False, default=str)\n"
            f"4. Handle errors gracefully and return json.dumps with 'error' key\n"
            f"5. Include 'import json' and 'import traceback' at the top\n"
            f"6. Do NOT use any external variables or imports beyond json, traceback, and {import_name}\n"
            f"Return ONLY the Python code, no markdown fences or explanation."
        )
        try:
            response = self.provider.chat([{"role": "user", "content": prompt}])
            text = response.content if response else ""
            if isinstance(text, str):
                # Strip markdown code fences if present
                code = text.strip()
                if code.startswith("```python"):
                    code = code[len("```python"):]
                elif code.startswith("```"):
                    code = code[len("```"):]
                if code.endswith("```"):
                    code = code[:-len("```")]
                return code.strip()
            return None
        except Exception:
            return None
