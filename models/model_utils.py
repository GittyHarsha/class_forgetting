import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
import torch 
import random

import torch

def get_mat_from_stacked(A, m, i):
    """
    Extracts a submatrix from a stacked matrix A based on m and index i.

    Parameters:
    A (torch.Tensor): The stacked matrix.
    m (list or torch.Tensor): List of row counts for each block.
    i (int): Index of the block to extract.

    Returns:
    tuple: A tuple containing:
           - Ai (torch.Tensor): The submatrix corresponding to block i.
           - row_inds (torch.Tensor): The indices of the rows corresponding to block i.
    """
    assert len(m) >= i, "Length of m must be at least i"

    row_start = sum(m[:i-1])
    row_end = sum(m[:i])
    row_inds = torch.arange(row_start, row_end)

    if A is None or A.numel() == 0:
        Ai = torch.empty((0, A.shape[1])) if A is not None else torch.empty((0, 0))
    else:
        Ai = A[row_inds, :]

    return Ai, row_inds

def parse_args(**kwargs):
    """
    Parses and validates input arguments with default values.

    Parameters:
    **kwargs: Arbitrary keyword arguments for parameter customization.

    Returns:
    dict: Parsed and validated arguments.
    """
    # Default values and checks
    default_args = {
        'RANK_TOL_A': 1e-14,  # Tolerance for rank calculation
        'ppi': 1e-3,          # Regularization parameter
        'ZEROTOL': 1e-14,     # Tolerance for singular value threshold
        'EPS_REL_ISO': 1e-6,  # Tolerance for isolated subspaces
        'DISABLE_WARNINGS': False  # Flag to disable warnings
    }
    check_POS = lambda x: x >= 0  # Non-negative check
    check_EPS = lambda x: 0 < x < 1  # Between 0 and 1 check

    # Validation
    for key, value in kwargs.items():
        if key in default_args:
            if key in ['ppi', 'RANK_TOL_A', 'ZEROTOL']:
                assert check_POS(value), f"Invalid value for {key}: {value}. Must be non-negative."
            if key == 'EPS_REL_ISO':
                assert check_EPS(value), f"Invalid value for {key}: {value}. Must be in the range (0, 1)."
        else:
            raise ValueError(f"Unknown argument: {key}")

        # Update default values with user-provided values
        default_args[key] = value

    return default_args

import torch

def hocsd(Q, m, **kwargs):
    """
    High-Order Canonical Singular Value Decomposition (HO-CSD) for rank-deficient matrices.

    This function computes the HO-CSD for N matrices, where Q is a vertically stacked
    matrix: Q = [Q1; Q2; ...; QN], with each Qi having m[i] rows and n columns,
    satisfying Q.T @ Q = I. It returns matrices U, S, Z, and eigenvalues Tau such that:

        Qi = Ui @ Si @ Z.T,
        Ui = U[sum(m[:i-1]):sum(m[:i]), :],
        Si = S[n*(i-1):n*i, :].

    Parameters:
    -----------
    Q : torch.Tensor
        A stacked matrix [Q1; Q2; ...; QN], with each Qi having m[i] rows and n columns.
    m : list
        A list of integers specifying the number of rows in each block Qi.
    **kwargs : dict, optional
        Keyword arguments for configuration:
        - ppi (float, default=1e-3): Regularization parameter.
        - ZEROTOL (float, default=1e-14): Tolerance for generalized singular values.
        - EPS_REL_ISO (float, default=1e-6): Tolerance for identifying isolated subspaces.
        - DISABLE_WARNINGS (bool, default=False): Flag to disable warnings.

    Returns:
    --------
    U : torch.Tensor
        A stacked matrix of orthogonal matrices [U1; U2; ...; UN].
    S : torch.Tensor
        A block diagonal matrix of singular values [S1; S2; ...; SN].
    Z : torch.Tensor
        A matrix of eigenvectors (Z.T @ Z = I).
    Tau : torch.Tensor
        A diagonal matrix of eigenvalues of T.
    taumin : float
        The theoretical minimum eigenvalue of Tau.
    taumax : float
        The theoretical maximum eigenvalue of Tau.
    iso_classes : list
        Indices of matrices Qi with unit generalized singular values.

    Notes:
    ------
    - The HO-CSD is applicable to rank-deficient matrices.
    - The input matrix Q must satisfy Q.T @ Q = I within tolerance ZEROTOL.
    - Optional keyword arguments allow control over tolerances and warnings.
    """
    # Hard-coded warnings thresholds
    WARN_EPS_ISO = 1e-6  # Warning tolerance for isolated subspace alignment
    WARN_COND = 1e6      # Warning threshold for condition number

    # Number of blocks (N) and columns (n) in Q
    N = len(m)  # Number of blocks
    n = Q.shape[1]  # Number of columns

    # Validate the dimensions of Q
    if sum(m) < n:
        raise ValueError(
            f"sum(m)={sum(m)} < n={n}. "
            f"Rank(Q)={n} is required."
        )
    if Q.shape[0] != sum(m):
        raise ValueError(
            f"Q.shape[0]={Q.shape[0]} does not match sum(m)={sum(m)}."
        )

    # Parse optional arguments
    args = parse_args(**kwargs)

    # Warning if Q'Q is not approximately identity
    if not args['DISABLE_WARNINGS']:
        deviation = torch.norm(Q.T @ Q - torch.eye(n)) / n
        if deviation >= args['ZEROTOL']:
            print(
                f"Warning: norm(Q.T @ Q - I)/n={deviation:.2e} "
                f"is greater than ZEROTOL={args['ZEROTOL']:.2e}."
            )

    # Initialize the Rhat matrix
    Rhat = torch.zeros((n, N * n))  # Rhat is (n x N*n)
    sqrt_ppi = torch.sqrt(torch.tensor(args['ppi']))  # Square root of the regularization parameter

    # Compute the blocks of Rhat
    for i in range(N):
        # Extract the block Qi
        Qi, _ = get_mat_from_stacked(Q, m, i + 1)

        # QR decomposition of [Qi; sqrt_ppi * I]
        _, Rhati = torch.linalg.qr(torch.vstack([Qi, sqrt_ppi * torch.eye(n)]))

        # Inverse of Rhati is stored in the appropriate block of Rhat
        Rhat[:, i * n:(i + 1) * n] = torch.linalg.inv(Rhati)

        # Check the condition number of Rhati and warn if necessary
        if not args['DISABLE_WARNINGS']:
            condition_number = torch.linalg.cond(Rhati)
            if condition_number >= WARN_COND:
                print(
                    f"Warning: For i={i + 1}, cond(Rhati)={condition_number:.2e} "
                    f"is greater than WARN_COND={WARN_COND}."
                )

    # Perform Singular Value Decomposition (SVD) on Rhat
    Z, sqrt_Tau, _ = torch.linalg.svd(Rhat, full_matrices=False)

    # Compute Tau (diagonal matrix of squared singular values, normalized by N)
    singular_values = sqrt_Tau  # Extract diagonal elements as 1D array
    Tau = torch.diag((singular_values ** 2) / N)

    # Theoretical minimum and maximum eigenvalues of Tau
    taumin = 1 / (1 / N + args['ppi'])  # Minimum eigenvalue
    taumax = (N - 1) / (N * args['ppi']) + 1 / (N * (1 + args['ppi']))  # Maximum eigenvalue

    # Identify indices corresponding to the isolated subspace
    ind_iso = torch.abs(taumax * torch.ones(len(singular_values)) - torch.diagonal(Tau)) <= (taumax - taumin) * args['EPS_REL_ISO']

    # Align eigenvectors associated with the isolated subspace
    iso_classes = []
    if torch.any(ind_iso):  # If there are isolated subspaces
        Z_iso = Z[:, ind_iso]  # Eigenvectors corresponding to the isolated subspace
        Z_iso_new = torch.zeros_like(Z_iso)  # Placeholder for aligned eigenvectors
        n_iso = torch.sum(ind_iso)  # Number of isolated subspaces
        iso_classes = torch.zeros(n_iso, dtype=torch.int)

        Z_iter = Z_iso.clone()
        for i in range(n_iso - 1):
            # Compute norms for all blocks to find the largest gain
            all_S = torch.zeros(N)
            for j in range(N):
                Qj, _ = get_mat_from_stacked(Q, m, j + 1)
                all_S[j] = torch.linalg.norm(Qj @ Z_iter, ord=2)

            # Sort blocks by their contribution
            ind_sorted = torch.argsort(-all_S)  # Descending order
            iso_classes[i] = ind_sorted[0]  # Store the class of the current isolated subspace

            # Align the isolated subspace
            Qiso, _ = get_mat_from_stacked(Q, m, iso_classes[i] + 1)
            _, _, Xiso = torch.linalg.svd(Qiso @ Z_iter)
            Z_iso_new[:, i] = Z_iter @ Xiso[:, 0]
            Z_iter = Z_iter @ Xiso[:, 1:]

        # Handle the final iteration (Z_iter is a vector)
        all_S = torch.zeros(N)
        for j in range(N):
            Qj, _ = get_mat_from_stacked(Q, m, j + 1)
            all_S[j] = torch.linalg.norm(Qj @ Z_iter, ord=2)

        ind_sorted = torch.argsort(-all_S)  # Descending order
        iso_classes[-1] = ind_sorted[0]
        Z_iso_new[:, -1] = Z_iter.squeeze()

        # Replace original Z_iso with the aligned version
        Z[:, ind_iso] = Z_iso_new

        # Verify orthogonality of the updated Z
        if not args['DISABLE_WARNINGS']:
            ortho_deviation = torch.linalg.norm(torch.eye(n) - Z.T @ Z, ord=2)
            if ortho_deviation > WARN_EPS_ISO:
                print(
                    f"Warning: Rotated Z is not orthogonal, "
                    f"norm(I - Z.T @ Z)={ortho_deviation:.2e}."
                )

    # Check if Z is orthogonal
    not_ortho = torch.linalg.norm(torch.eye(n) - Z.T @ Z, ord=2) > WARN_EPS_ISO

    # Initialize S and U matrices
    S = torch.zeros((N * n, n))  # Block diagonal matrix for singular values
    U = torch.zeros((sum(m), n))  # Stacked orthogonal matrix

    # Compute U and S for each block Qi
    for i in range(N):
        Qi, _ = get_mat_from_stacked(Q, m, i + 1)

        # Adjust based on whether Z is orthogonal
        if not_ortho:
            Bi = Qi @ torch.linalg.inv(Z.T)  # Z is not orthogonal
        else:
            Bi = Qi @ Z  # Z is orthogonal

        # Compute singular values (column norms of Bi)
        Si = torch.linalg.norm(Bi, dim=0)

        # Indices for significant singular values
        ind_pos = Si > args['ZEROTOL']

        # Compute U for nonzero singular values
        U[sum(m[:i]):sum(m[:i+1]), ind_pos] = Bi[:, ind_pos] / Si[ind_pos]

        # Process columns with zero singular values (numerical rank deficiency)
        nzero = torch.sum(~ind_pos)
        if nzero > 0:
            UQi, SQi, _ = torch.linalg.svd(Qi)
            ind_zero_i = SQi <= args['ZEROTOL']
            ni2 = torch.sum(ind_zero_i)

            if ni2 == 0:
                # Normalize columns of Qi for zero singular values
                Qitmp = Qi[:, ~ind_pos]
                Qitmp_norm = torch.linalg.norm(Qitmp, dim=0)
                Qitmp_norm[Qitmp_norm <= args['ZEROTOL']] = 1
                U[sum(m[:i]):sum(m[:i+1]), ~ind_pos] = Qitmp / Qitmp_norm
            else:
                # Use right singular vectors associated with zero singular values
                Ui2 = UQi[:, ind_zero_i]
                if ni2 < nzero:
                    Ui2 = Ui2.repeat(1, int(torch.ceil(nzero / ni2)))
                U[sum(m[:i]):sum(m[:i+1]), ~ind_pos] = Ui2[:, :nzero]

        # Store singular values in S
        S[i * n:(i + 1) * n, :] = torch.diag(Si)

    # Verify reconstruction of Qi
    if not args['DISABLE_WARNINGS']:
        for i in range(N):
            Ui, _ = get_mat_from_stacked(U, m, i + 1)
            Si, _ = get_mat_from_stacked(S, [n] * N, i + 1)
            Qi, _ = get_mat_from_stacked(Q, m, i + 1)
            reconstruction_error = torch.sum(torch.abs(Ui @ Si @ Z.T - Qi))
            if reconstruction_error >= 1e-12:
                print(
                    f"Warning: Reconstruction error={reconstruction_error:.2e} "
                    f"for matrix {i + 1}."
                )
    return U, S, Z, Tau, taumin, taumax, iso_classes


import torch

def hogsvd(A, m, **kwargs):
    """
    High-Order Generalized Singular Value Decomposition (HO-GSVD).

    This function computes the HO-GSVD for N matrices with A = [A1; A2; ...; AN],
    where each Ai has m[i] rows and n columns. The result satisfies:
        Ai = Ui @ Si @ V.T,
    where Ui, Si are the components for each block, and V is shared.

    Parameters:
    -----------
    A : torch.Tensor
        A stacked matrix [A1; A2; ...; AN] of size (sum(m), n).
    m : list
        A list of integers specifying the number of rows in each block Ai.
    **kwargs : dict, optional
        Additional parameters:
        - RANK_TOL_A (float, default=1e-14): Rank tolerance for padding A.
        - ppi (float, default=1e-3): See `hocsd`.
        - ZEROTOL (float, default=1e-14): See `hocsd`.
        - EPS_REL_ISO (float, default=1e-6): See `hocsd`.
        - DISABLE_WARNINGS (bool, default=False): Disable warnings.

    Returns:
    --------
    U : torch.Tensor
        A stacked orthogonal matrix [U1; U2; ...; UN].
    S : torch.Tensor
        A block-diagonal matrix [S1; S2; ...; SN] of singular values.
    V : torch.Tensor
        A shared orthogonal matrix of size (n, n).
    Tau : torch.Tensor
        Eigenvalues of the matrix T used in HO-CSD.
    taumin : float
        Minimum theoretical eigenvalue of Tau.
    taumax : float
        Maximum theoretical eigenvalue of Tau.
    iso_classes : list
        Indices of matrices Ai with isolated subspaces.
    """
    n = A.shape[1]  # Number of columns
    N = len(m)      # Number of blocks
    if A.shape[0] != sum(m):
        raise ValueError(f"A.shape[0] = {A.shape[0]} does not match sum(m) = {sum(m)}.")

    # Parse optional arguments
    args = parse_args(**kwargs)

    # Compute the singular values of A
    SA = torch.linalg.svdvals(A)
    rank_def_A = n - torch.sum(SA > args['RANK_TOL_A'])

    if rank_def_A == 0:
        Apad = A
    else:
        if not args['DISABLE_WARNINGS']:
            print(
                f"Warning: Provided rank-deficient A with rank(A)={n - rank_def_A} < n={n}. "
                f"Padding A."
            )

        # Compute SVD of A and use the last singular vectors for padding
        _, _, VA = torch.linalg.svd(A)
        padding = VA[:, -rank_def_A:].T  # Rows to pad
        Apad = torch.vstack([A, padding])
        m.append(rank_def_A)  # Extend m to account for padding

    # Perform QR decomposition
    Q, R = torch.linalg.qr(Apad, mode='reduced')

    # Call hocsd with Q and the updated m
    U, S, Z, Tau, taumin, taumax, iso_classes = hocsd(Q, m, **kwargs)

    # Compute V from R and Z
    V = R.T @ Z

    # Remove padding if rank_def_A > 0
    if rank_def_A > 0:
        U = U[:-rank_def_A, :]  # Remove rows added for padding
        S = S[:-(n - rank_def_A), :]  # Adjust S to exclude padding

    return U, S, V, Tau, taumin, taumax, iso_classes


def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin[0]+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/float(stride[0])+1)), int(np.floor((Lin[1]+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/float(stride[1])+1))


def reshape_conv_input_activation(x, conv_layer=None, kernel_size=3, stride=1, padding=0, dilation=1, max_samples=10000):
    ### FAST CODE (Avoid for loops)
    if conv_layer:
        kernel_size = conv_layer.kernel_size
        stride = conv_layer.stride
        padding =  conv_layer.padding 
        dilation = conv_layer.dilation
    if x.shape[-1] > 3*kernel_size[-1]:
        start_index_i =random.randint(0, x.shape[-1]-3*kernel_size[-1])
        start_index_j =random.randint(0, x.shape[-2]-3*kernel_size[-2])
        sampled_x = x[:,:,start_index_i:start_index_i+3*kernel_size[-2],start_index_j:start_index_j+3*kernel_size[-1] ]
        x_unfold = torch.nn.functional.unfold(sampled_x, kernel_size, dilation=dilation, padding=padding, stride=stride)
    else:
        x_unfold = torch.nn.functional.unfold(x, kernel_size, dilation=dilation, padding=padding, stride=stride)
    mat = x_unfold.permute(0,2,1).contiguous().view(-1,x_unfold.shape[1])
    r=np.arange(mat.shape[0])
    np.random.shuffle(r)
    b = r[:max_samples]
    mat = mat[b]
    return mat

def forward_cache_activations(x, layer, key, max_samples=10000):
    act=OrderedDict()  
    if isinstance(layer, nn.Conv2d):
        act[key]=reshape_conv_input_activation(x.clone().detach(), layer,max_samples =  max_samples)
        x = layer(x)
    elif isinstance(layer, nn.Linear):
        act[key]=x.clone().detach()
        x = layer(x)
    else:
        x = layer(x)
    return x, act 



def forward_cache_projections(x, layer, key, alpha, max_samples=10000):
    Proj=OrderedDict()  
    if isinstance(layer, nn.Conv2d):
        activation = reshape_conv_input_activation(x.clone().detach(), layer, max_samples =  max_samples).transpose(0,1)
        Ur,Sr,_ = torch.linalg.svd(activation, full_matrices=False)
        sval_total = (Sr**2).sum()
        sval_ratio = (Sr**2)/sval_total
        importance_r =  torch.diag( alpha *sval_ratio/((alpha-1)*sval_ratio+1) )
        mr = torch.mm( Ur, importance_r )
        Proj[key] =  torch.mm( mr, Ur.transpose(0,1) )
        x = layer(x)
    elif isinstance(layer, nn.Linear):
        activation = x.clone().detach().transpose(0,1)
        Ur,Sr,_ = torch.linalg.svd(activation, full_matrices=False)
        sval_total = (Sr**2).sum()
        sval_ratio = (Sr**2)/sval_total
        importance_r =  torch.diag( alpha *sval_ratio/((alpha-1)*sval_ratio+1) )
        mr = torch.mm( Ur, importance_r )
        Proj[key] = torch.mm( mr, Ur.transpose(0,1) )
        x = layer(x)
    else:
        x = layer(x)
    
    
    return x, Proj 

def forward_cache_svd(x, layer, key,  max_samples=10000):
    U = OrderedDict()  
    S = OrderedDict()  
    if isinstance(layer, nn.Conv2d):
        activation = reshape_conv_input_activation(x.clone().detach(), layer, max_samples =  max_samples).transpose(0,1)
        Ur,Sr,_ = torch.linalg.svd(activation, full_matrices=False)
        U[key] = Ur
        S[key] = Sr
        x = layer(x)
    elif isinstance(layer, nn.Linear):
        activation = x.clone().detach().transpose(0,1)
        Ur,Sr,_ = torch.linalg.svd(activation, full_matrices=False)
        U[key] = Ur
        S[key] = Sr
        x = layer(x)
    else:
        x = layer(x)
    
    
    return x, U, S 


def forward_cache_hogsvd():
    pass

def forward_cache_projections_hogsvd():
    pass