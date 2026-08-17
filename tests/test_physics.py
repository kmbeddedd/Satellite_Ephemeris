import numpy as np

from src.physics import ecef_error_to_ric, ric_basis, ric_error_to_ecef


def test_ric_basis_is_orthonormal_and_round_trips():
    position = np.array([[26_000_000.0, 1_000_000.0, 4_000_000.0]])
    velocity = np.array([[-200.0, 3_000.0, 600.0]])
    error = np.array([[2.0, -3.0, 5.0]])
    basis = ric_basis(position, velocity)
    identity = np.einsum("...ji,...jk->...ik", basis, basis)
    np.testing.assert_allclose(identity, np.eye(3)[None], atol=1e-12)
    ric = ecef_error_to_ric(error, position, velocity)
    np.testing.assert_allclose(ric_error_to_ecef(ric, position, velocity), error, atol=1e-12)

