import torch

def batch_concatenate_quaternions(Q1, Q2, out=None):
    """Concatenate two batches of quaternions using Hamilton's quaternion multiplication.

    Parameters
    ----------
    Q1 : torch.Tensor, shape (..., 4)
        First batch of quaternions
    Q2 : torch.Tensor, shape (..., 4)
        Second batch of quaternions
    out : torch.Tensor, shape (..., 4), optional (default: new tensor)
        Output tensor to which we write the result

    Returns
    -------
    Q12 : torch.Tensor, shape (..., 4)
        Batch of quaternions that represents the concatenated rotations
    """
    # Ensure input tensors are of the correct shape
    if Q1.ndimension() != Q2.ndimension():
        raise ValueError(f"Number of dimensions must be the same. Got {Q1.ndimension()} for Q1 and {Q2.ndimension()} for Q2.")

    for d in range(Q1.ndimension() - 1):  # Exclude the last dimension (which is the quaternion dimension)
        if Q1.shape[d] != Q2.shape[d]:
            raise ValueError(f"Size of dimension {d + 1} does not match: {Q1.shape[d]} != {Q2.shape[d]}")

    if Q1.shape[-1] != 4 or Q2.shape[-1] != 4:
        raise ValueError(f"Last dimension must be of size 4. Got {Q1.shape[-1]} and {Q2.shape[-1]}.")

    if out is None:
        out = torch.empty_like(Q1)

    # Compute the scalar part (real component) of the quaternion multiplication
    scalar_part = Q1[..., 0] * Q2[..., 0] - torch.sum(Q1[..., 1:] * Q2[..., 1:], dim=-1)
    out[..., 0] = scalar_part

    # Compute the vector part (imaginary component) of the quaternion multiplication
    vector_part = (Q1[..., 0:1] * Q2[..., 1:] +  # (w1 * v2)
                   Q2[..., 0:1] * Q1[..., 1:] +  # (w2 * v1)
                   torch.cross(Q1[..., 1:], Q2[..., 1:], dim=-1))  # (v1 x v2)

    out[..., 1:] = vector_part
    return out


def batch_q_conj(Q):
    """Conjugate of quaternions.

    The conjugate of a unit quaternion inverts the rotation represented by
    this unit quaternion. The conjugate of a quaternion q is often denoted
    as q*.

    Parameters
    ----------
    Q : torch.Tensor, shape (..., 4)
        Unit quaternions to represent rotations: (w, x, y, z)

    Returns
    -------
    Q_c : torch.Tensor, shape (..., 4)
        Conjugates (w, -x, -y, -z)
    """
    # Ensure input tensor is a PyTorch tensor
    Q = torch.tensor(Q) if not isinstance(Q, torch.Tensor) else Q

    # Initialize output tensor of the same shape as Q
    out = torch.empty_like(Q)

    # Set the real part (w) to be the same as in Q
    out[..., 0] = Q[..., 0]

    # Set the imaginary parts (-x, -y, -z) to be the negative values of Q
    out[..., 1:] = -Q[..., 1:]

    return out


def batch_log_quaternion(q):
    """Compute the logarithm of a quaternion.

    Args:
        q (torch.Tensor): Quaternion (w, x, y, z), shape (..., 4)

    Returns:
        torch.Tensor: Logarithm of quaternion, shape (..., 4)
    """
    # Ensure input is a tensor
    q = torch.tensor(q) if not isinstance(q, torch.Tensor) else q

    # Norm of the quaternion (if it's not already normalized)
    norm_q = torch.norm(q, dim=-1, keepdim=True)

    # Compute the angle theta
    theta = torch.acos(q[:, 0] / norm_q.squeeze())  # q[:, 0] is the scalar part (w)
    theta = theta.unsqueeze(-1)

    # Compute the vector part (x, y, z)
    vector_part = q[:, 1:] / torch.sin(theta)  # (x, y, z) normalized

    # Compute the logarithm
    log_q = torch.cat([torch.log(norm_q), theta * vector_part], dim=-1)

    return log_q

