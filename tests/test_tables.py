"""What TabICL expects of a table, and how it is met."""
import math

import torch

from table_image_icl.tables import contiguous_codes, prepare_table

N_CTX = 8


def test_contiguous_codes_number_the_classes_of_each_table_from_zero_and_remember_the_slots():
    y = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1], [5, 6, 5, 6, 5, 6, 5, 6]])
    codes, slots = contiguous_codes(y, n_slots=10)
    assert codes.tolist() == [[0, 1, 2, 0, 1, 2, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1]]
    assert slots[0, :4].tolist() == [0, 1, 2, -1] and slots[1, :3].tolist() == [5, 6, -1]


def test_prepare_table_standardises_on_the_context_fills_missing_cells_and_softens_outliers():
    x = torch.randn(2, N_CTX + 4, 3, generator=torch.Generator().manual_seed(0))
    x[0, N_CTX + 1, 0] = float("nan")                                        # a missing test cell
    x[1, 2, 1] = 1e4                                                         # a wild context value
    z = prepare_table(x, N_CTX)
    assert torch.isfinite(z).all()
    assert z[0, N_CTX + 1, 0] == 0.0, "missing = the context mean, which is 0 after standardising"
    context = z[0, :N_CTX, 2]
    assert abs(context.mean()) < 1e-5 and abs(context.std(correction=0) - 1) < 1e-3
    assert z[1, 2, 1] < 4.0 + math.log1p(1e4), "clipped softly at four standard deviations"
    assert torch.allclose(prepare_table(z, N_CTX)[0, :, 2], z[0, :, 2], atol=1e-4), "a standard column passes through"


def test_test_rows_do_not_change_the_scale():
    x = torch.randn(1, N_CTX + 4, 2, generator=torch.Generator().manual_seed(0))
    z = prepare_table(x, N_CTX)
    x[0, N_CTX:] *= 100
    assert torch.allclose(prepare_table(x, N_CTX)[0, :N_CTX], z[0, :N_CTX])
