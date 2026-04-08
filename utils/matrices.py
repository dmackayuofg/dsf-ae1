
# ===============================================================
# matrices.py — Final unified version (Style A + GLOBAL alignment)
# ===============================================================

import numpy as np
import IPython.display as disp

# Optional SymPy
try:
    import sympy
    sympy.init_printing(use_latex='mathjax')
except Exception:
    sympy = False


# ===============================================================
# 1. Matrix Printer
# ===============================================================

def print_matrix(name, matrix, prec=2):
    if not hasattr(matrix, "__len__"):
        disp.display(disp.Latex(f"${name} = {matrix}$"))
        return

    if isinstance(matrix, list):
        matrix = np.array(matrix)

    if hasattr(matrix, "dtype") and str(matrix.dtype).startswith("float"):
        matrix = np.around(matrix, prec)

    if matrix.ndim == 1:
        matrix = matrix[None, :]

    matrix_list = matrix.tolist()

    if sympy:
        M = sympy.Matrix(matrix_list)
        disp.display(disp.Latex(f"${name} = {sympy.latex(M)}$"))
    else:
        print(name, "\n", matrix)



# ===============================================================
# 2. GLOBAL‑WIDTH TENSOR PRINTER
# ===============================================================

def _latex_row(items):
    """Format LaTeX row, wrapping each item in braces."""
    return " & ".join([f"{{{item}}}" for item in items])


def make_boxed_tensor_latex(x, _global_width=None):
    """
    GLOBAL‑WIDTH VERSION.
    Style A:
      • 2D tensors: rows boxed, stacked vertically, whole tensor boxed.
      • Higher dimensions: recursively a matrix of subtensors.
    All scalar prints at the same recursion depth use the SAME width.
    """

    x = np.asarray(x)

    # ----- Determine global width for this depth -----
    if _global_width is None:
        # width = max # of digits across the entire tensor at this depth
        _global_width = max(len(str(int(v))) for v in x.flatten())

    # ----- Ensure at least 2D -----
    if x.ndim == 1:
        x = x[None, :]

    rows, cols = x.shape[:2]
    latex_rows = []

    # ----- Build each row -----
    for r in range(rows):
        row_items = []

        for c in range(cols):
            if x.ndim == 2:
                # pad to global width
                s = str(int(x[r, c]))
                pad = _global_width - len(s)
                row_items.append(r"\phantom{" + "0"*pad + "}" + s)

            else:
                # recursive subtensor uses SAME global width
                row_items.append(make_boxed_tensor_latex(
                    x[r, c],
                    _global_width=_global_width
                ))

        # ------ BOX THE ROW using internal array (ensures alignment) ------
        col_spec = "c" * len(row_items)
        latex_rows.append(
            r"\boxed{"
            r"\begin{array}{" + col_spec + "}"
            + _latex_row(row_items) +
            r"\end{array}"
            r"}"
        )

    # ----- STACK ROWS VERTICALLY -----
    body = r" \\ ".join(latex_rows)

    # ----- OUTER BOX -----
    return (
        r"\boxed{\begin{array}{c}"
        + body +
        r"\end{array}}"
    )


def show_boxed_tensor_latex(x):
    disp.display(disp.Latex(r"$$" + make_boxed_tensor_latex(x) + r"$$"))



# ===============================================================
# 3. ASCII Printer
# ===============================================================

def _boxed_tensor_ascii(x):
    if np.isscalar(x):
        return [f"{int(x):4d}"]

    x = np.asarray(x)

    if len(x.shape) & 1 == 1:
        x = x.reshape(tuple(list(x.shape) + [1]))

    rows, cols = x.shape[:2]
    assembled = []

    for r in range(rows):
        blocks = [_boxed_tensor_ascii(x[r, c]) for c in range(cols)]
        height = max(len(b) for b in blocks)

        for h in range(height):
            parts = [(b[h] if h < len(b) else " " * len(b[0])) for b in blocks]
            assembled.append(" ".join(parts))

        assembled.append("")

    return assembled


def boxed_tensor_ascii(x):
    return "\n".join(_boxed_tensor_ascii(x))


# ===============================================================
# END OF FILE
# ===============================================================

# # ===============================================================
# # matrices.py — Clean unified version (Style A + padding)
# # ===============================================================

# import numpy as np
# import IPython.display as disp

# # Optional SymPy support
# try:
#     import sympy
#     sympy.init_printing(use_latex='mathjax')
# except Exception:
#     sympy = False


# # ===============================================================
# # 1. Matrix Printer
# # ===============================================================

# def print_matrix(name, matrix, prec=2):
#     """Pretty-print a matrix using LaTeX (via SymPy)."""

#     if not hasattr(matrix, "__len__"):
#         disp.display(disp.Latex(f"${name} = {matrix}$"))
#         return

#     if isinstance(matrix, list):
#         matrix = np.array(matrix)

#     if hasattr(matrix, "dtype") and str(matrix.dtype).startswith("float"):
#         matrix = np.around(matrix, prec)

#     if matrix.ndim == 1:
#         matrix = matrix[None, :]

#     matrix_list = matrix.tolist()

#     if sympy:
#         M = sympy.Matrix(matrix_list)
#         disp.display(disp.Latex(f"${name} = {sympy.latex(M)}$"))
#     else:
#         print(name, "\n", matrix)


# # ===============================================================
# # 2. MathJax‑Safe LaTeX Tensor Printer (Style A + alignment padding)
# # ===============================================================

# def _latex_padded_number(n, width):
#     l=width-len(str(n))
#     """Pad number to fixed width using phantom digits for alignment."""
#     return r"\phantom{" + "0"*l + "}" + str(int(n))


# def _latex_row(items):
#     """Format a horizontal row inside an array; wrap items in braces."""
#     return " & ".join([f"{{{item}}}" for item in items])


# def make_boxed_tensor_latex(x, box_rows=True):
#     """
#     Recursively format a tensor into nested boxed LaTeX.

#     Style A:
#       • For 2D tensors: each row boxed, stacked vertically, entire tensor boxed.
#       • For higher dimensions: tensor is matrix of subtensors.
#     """

#     x = np.asarray(x)

#     # ensure at least 2D
#     if x.ndim == 1:
#         x = x[None, :]

#     rows, cols = x.shape[:2]
#     latex_rows = []

#     # compute max digit width for *this tensor block* (local formatting)
#     max_width = max(len(str(int(v))) for v in x.flatten())

#     for r in range(rows):
#         row_items = []
#         for c in range(cols):

#             if x.ndim == 2:
#                 # pad each number to same width
#                 row_items.append(_latex_padded_number(x[r, c], max_width))

#             else:
#                 # recurse for subtensors
#                 row_items.append(make_boxed_tensor_latex(x[r, c], box_rows=True))

#         # Box THIS row (MathJax-safe: row is an array in a box)
#         num_cols = len(row_items)
#         col_spec = "c" * num_cols

#         latex_rows.append(
#             r"\boxed{\begin{array}{" + col_spec + "}" +
#             _latex_row(row_items) +
#             r"\end{array}}"
#         )

#     # Stack rows vertically in a 1-column array
#     body = r" \\ ".join(latex_rows)

#     # Box the entire tensor
#     return (
#         r"\boxed{\begin{array}{c}"
#         + body +
#         r"\end{array}}"
#     )


# def show_boxed_tensor_latex(x, box_rows=True):
#     """Display boxed tensor in Jupyter."""
#     code = make_boxed_tensor_latex(x, box_rows=box_rows)
#     disp.display(disp.Latex(r"\[" + code + r"\]"))


# # ===============================================================
# # 3. ASCII Tensor Printer
# # ===============================================================

# def _boxed_tensor_ascii(x):
#     if np.isscalar(x):
#         return [f"{int(x):4d}"]

#     x = np.asarray(x)

#     if len(x.shape) & 1 == 1:
#         x = x.reshape(tuple(list(x.shape) + [1]))

#     rows, cols = x.shape[0:2]
#     assembled = []

#     for r in range(rows):
#         blocks = [_boxed_tensor_ascii(x[r, c]) for c in range(cols)]
#         height = max(len(b) for b in blocks)

#         for h in range(height):
#             parts = [(b[h] if h < len(b) else " " * len(b[0])) for b in blocks]
#             assembled.append(" ".join(parts))

#         assembled.append("")

#     return assembled


# def boxed_tensor_ascii(x):
#     return "\n".join(_boxed_tensor_ascii(x))


# # ===============================================================
# # END OF FILE
# # ===============================================================
