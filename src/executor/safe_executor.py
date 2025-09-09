import pandas as pd

# Whitelist of allowed functions/keywords
SAFE_GLOBALS = {
    "pd": pd
}

def execute_query(code: str, df: pd.DataFrame):
    """
    Execute the Gemini-generated code safely on the provided DataFrame.
    Returns result as string.
    """
    # Prepare local environment
    local_env = {"df": df.copy(), "result": None}

    try:
        # Only allow pd and df, no os, open, subprocess, etc.
        exec(code, SAFE_GLOBALS, local_env)

        # Expect Gemini to assign final output to 'result'
        if "result" in local_env:
            res = local_env["result"]
            # Convert DataFrame or Series to string for CLI display
            if isinstance(res, pd.DataFrame) or isinstance(res, pd.Series):
                return res.to_string(index=False)
            else:
                return str(res)
        else:
            return "Query executed, but no 'result' variable returned by Gemini."

    except Exception as e:
        return f"⚠ Could not process the query. Error: {e}"