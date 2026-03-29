This directory contains raw profile-matrix analysis tools for the NKalpha
direct-democracy setting.

The main script is:

`analyze_raw_profile_matrices.py`

It samples raw voter-by-candidate utility matrices `U` along the fixed
iso-complexity curves already used in the paper, computes matrix-level feature
vectors, and saves both the raw matrices and derived diagnostics for later
inspection or clustering.
