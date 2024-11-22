import torch
import triton
import triton.language as tl

from logging import getLogger
logger = getLogger()

@triton.jit
def _rms_norm_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    # Compute the squared sum of the input values
    _sq_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _sq_sum += x * x

    sq_sum = tl.sum(_sq_sum, axis=0) / N  # mean of squares
    rnorm = 1 / tl.sqrt(sq_sum + eps)  # root mean square inverse

    # Normalize and apply scaling
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        x_hat = x * rnorm
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y, mask=mask)


def rms_norm(x, weight, eps):
    # Flatten all dimensions except the last one
    num_rows = x.shape[:-1].numel()  # total number of "rows" when flattening all but the last dimension
    N = x.shape[-1]  # size of the last dimension (columns)

    # Reshape the tensor to 2D (flattening all except the last dimension)
    x_flat = x.view(-1, N)
    y_flat = torch.empty_like(x_flat)
    logger.debug(f"rms_norm: {num_rows}x{N}")

    BLOCK_SIZE = 1024

    # Calculate the effective stride, which is the distance between consecutive rows in the flattened view.
    stride = x.stride(-2) if len(x.shape) > 1 else N

    # Launch the Triton kernel with the flattened tensor
    _rms_norm_kernel[(num_rows,)](x_flat, y_flat, weight, stride, N, eps, BLOCK_SIZE)

    # Reshape the result back to the original tensor shape
    return y_flat.view_as(x)


# From https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
@triton.jit
def _matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,  #
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FUSED_ADD: tl.constexpr,
    FUSED_DIV: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if FUSED_DIV != 0:
        accumulator = tl.fdiv(accumulator, FUSED_DIV)
    if FUSED_ADD:
        d_ptrs = d_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        d = tl.load(d_ptrs, mask=c_mask)
        accumulator += d
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b, d=None, fused_div=0):
    # Check constraints.
    assert a.shape[-1] == b.shape[-2], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert a.dtype == b.dtype, "Input types must match"

    # Flatten all leading dimensions into one batch dimension
    batch_dims = a.shape[:-2]  # Shape of leading dimensions (e.g., batch dimensions)
    M, K = a.shape[-2:]        # Last two dimensions of 'a' are M and K
    N = b.shape[-1]            # Last dimension of 'b' is N

    # Reshape inputs to handle batch dimension (collapse into 2D matrices for matmul)
    a_reshaped = a.reshape(-1, M, K)
    b_reshaped = b.reshape(-1, K, N)

    # Allocates output
    c_reshaped = torch.empty(a_reshaped.shape[:-2] + (M, N), device=a.device, dtype=a.dtype)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    # Loop through batch dimension
    for batch_idx in range(a_reshaped.shape[0]):
        logger.debug(f"matmul: {batch_idx}: {a_reshaped.shape[0]}")
        # Get current slices for batch
        a_slice = a_reshaped[batch_idx]
        b_slice = b_reshaped
        d_slice = d[batch_idx] if d is not None else None

        _matmul_kernel[grid](
            a_slice,
            b_slice,
            c_reshaped[batch_idx],  #
            d_slice,
            M,
            N,
            K,  #
            a_slice.stride(0),
            a_slice.stride(1),  #
            b_slice.stride(0),
            b_slice.stride(1),  #
            c_reshaped.stride(0),
            c_reshaped.stride(1),  #
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=128,
            GROUP_SIZE_M=4,
            FUSED_ADD=True if d_slice is not None else False,
            FUSED_DIV=fused_div,
        )

    # Reshape the output back to the original batch dimensions
    c = c_reshaped.view(batch_dims + (M, N))

    return c

# From https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
@triton.jit
def _softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    logger.debug(f"softmax: {n_rows}x{n_cols}")

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_stages = 4
    y = torch.empty_like(x)
    num_programs = min(num_programs, n_rows)
    _softmax_kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
    )
