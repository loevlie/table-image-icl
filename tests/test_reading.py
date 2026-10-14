"""The labelled read: what is read back is labels, who reads from whom, and the top-k retrieval."""
import inspect

import torch
import torch.nn.functional as F

from table_image_icl.reading import LabelRead, LabelledRead, nearest_photos, neighbour_mask

D, P, N_CTX, N_TEST, N_SLOTS = 16, 4, 6, 3, 8
Y = torch.tensor([[0, 1, 2, 0, 1, 2], [3, 4, 3, 4, 3, 4]])


def tokens(seed=0, batch=2, n=N_CTX + N_TEST):
    return torch.randn(batch, n, P, D, generator=torch.Generator().manual_seed(seed))


def stage(seed=0, max_sources=256, read="all", top_k=32):
    torch.manual_seed(seed)
    return LabelledRead(D, heads=2, layers=2, n_slots=N_SLOTS, max_sources=max_sources, read=read, top_k=top_k).eval()


def test_the_stage_keeps_the_shape_and_takes_only_context_labels():
    assert stage()(tokens(), Y).shape == (2, N_CTX + N_TEST, P, D)
    assert list(inspect.signature(LabelledRead.forward).parameters) == ["self", "tokens", "y_context"]


def test_a_test_photo_influences_only_itself():
    s, t = stage(), tokens()
    t2 = t.clone()
    t2[:, N_CTX + 1] = torch.randn(P, D)
    a, b = s(t, Y), s(t2, Y)
    others = [i for i in range(N_CTX + N_TEST) if i != N_CTX + 1]
    assert torch.allclose(a[:, others], b[:, others], atol=1e-6) and not torch.allclose(a[:, N_CTX + 1], b[:, N_CTX + 1])


def test_every_context_label_reaches_the_test_photos():
    s, t = stage(), tokens()
    base = s(t, Y)
    for j in range(N_CTX):
        y2 = Y.clone()
        y2[:, j] = (Y[:, j] + 1) % N_SLOTS
        assert not torch.allclose(base[:, N_CTX:], s(t, y2)[:, N_CTX:]), f"context label {j} is ignored"


def test_a_context_photo_reads_its_own_label_too():
    s, t = stage(), tokens()
    y2 = Y.clone()
    y2[:, 0] = (Y[:, 0] + 1) % N_SLOTS
    assert not torch.allclose(s(t, Y)[:, 0], s(t, y2)[:, 0])


def test_what_is_read_back_is_labels_not_content():
    """If every context photo has the same label, a test photo reads the same thing whatever the context shows."""
    s = stage()
    same = torch.zeros(2, N_CTX, dtype=torch.long)
    t, t2 = tokens(0), tokens(0)
    t2[:, :N_CTX] = tokens(1)[:, :N_CTX]
    assert torch.allclose(s(t, same)[:, N_CTX:], s(t2, same)[:, N_CTX:], atol=1e-5)


def test_at_most_max_sources_context_photos_are_read():
    s, t = stage(max_sources=4), tokens()
    base = s(t, Y)
    y2 = Y.clone()
    y2[:, 4:] = (Y[:, 4:] + 1) % N_SLOTS                         # context photos 4 and 5 are beyond the cap
    assert torch.allclose(base[:, N_CTX:], s(t, y2)[:, N_CTX:], atol=1e-6)
    s.train()
    picks = {tuple(sorted(s.pick_sources(N_CTX, "cpu").tolist())) for _ in range(30)}
    assert len(picks) > 1 and all(len(p) == 4 for p in picks), "training draws a new subset each time"


def test_the_source_cap_is_a_setting_not_a_weight():
    s = stage(max_sources=4)
    assert s.pick_sources(N_CTX, "cpu").tolist() == [0, 1, 2, 3]
    s.max_sources = 1024
    assert s.pick_sources(N_CTX, "cpu").tolist() == list(range(N_CTX))


def test_gradients_reach_every_parameter():
    s = stage().train()
    s(tokens(), Y).square().sum().backward()
    for name, p in s.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_the_stage_learns_where_to_look_from_the_labelled_context():
    """Each task draws new class markers; a photo of class c hides marker c in one random patch. Which marker means
    which class is known only from the labelled context photos."""
    torch.manual_seed(0)
    d, patches, n_ctx, n_test = 32, 5, 16, 8
    s = LabelledRead(d, heads=4, layers=2, n_slots=2, max_sources=256, read="all")
    head = torch.nn.Linear(d, 1)
    opt = torch.optim.Adam(list(s.parameters()) + list(head.parameters()), lr=1e-3)
    g = torch.Generator().manual_seed(1)

    def task(batch=4):
        markers = F.normalize(torch.randn(batch, 2, d, generator=g), dim=-1) * 6
        y = torch.randint(0, 2, (batch, n_ctx + n_test), generator=g)
        x = torch.randn(batch, n_ctx + n_test, patches, d, generator=g)
        where = torch.randint(0, patches, (batch, n_ctx + n_test), generator=g)
        index = where[..., None, None].expand(-1, -1, 1, d)
        x.scatter_add_(2, index, markers.gather(1, y[..., None].expand(-1, -1, d))[:, :, None])
        return x, y[:, :n_ctx], y[:, n_ctx:]

    def logits(x, y_ctx):
        return head(s(x, y_ctx)[:, n_ctx:].amax(2))[..., 0]

    for _ in range(1500):
        x, y_ctx, y_test = task()
        loss = F.binary_cross_entropy_with_logits(logits(x, y_ctx), y_test.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x, y_ctx, y_test = task(batch=64)
        accuracy = ((logits(x, y_ctx) > 0) == y_test.bool()).float().mean().item()
    assert accuracy > 0.75, accuracy


# ---------- top-k: retrieve, then read ----------

def test_nearest_photos_are_the_most_similar_by_mean_token():
    t = torch.zeros(1, 5, P, D)
    for i in range(5):
        t[0, i, :, i] = 1.0                                          # photo i points along axis i
    t[0, 4] = t[0, 1] * 0.9 + 0.01                                   # photo 4 looks like photo 1
    nearest = nearest_photos(t, t[:, :4], k=2)
    assert nearest.shape == (1, 5, 2)
    assert nearest[0, 4, 0] == 1, "the test photo's nearest context photo is the one it resembles"
    assert all(nearest[0, i, 0] == i for i in range(4)), "a context photo is its own nearest source"


def test_top_k_reads_only_from_the_nearest_context_photos():
    s, t = stage(read="top_k", top_k=2), tokens()
    nearest = nearest_photos(t, t[:, :N_CTX], k=2)[:, N_CTX:]
    base = s(t, Y)
    for j in range(N_CTX):
        y2 = Y.clone()
        y2[:, j] = (Y[:, j] + 1) % N_SLOTS
        changed = ~torch.isclose(s(t, y2)[:, N_CTX:], base[:, N_CTX:], atol=1e-6).all(-1).all(-1)
        assert torch.equal(changed, (nearest == j).any(-1)), f"context photo {j}'s label reaches exactly the test photos that retrieved it"


def test_top_k_with_every_context_photo_is_the_plain_read():
    plain, retrieving = stage(read="all"), stage(read="top_k", top_k=N_CTX)
    retrieving.load_state_dict(plain.state_dict())
    assert torch.allclose(plain(tokens(), Y), retrieving(tokens(), Y), atol=1e-5)


def test_top_k_reads_do_not_depend_on_the_chunk_of_photos():
    s, t = stage(read="top_k", top_k=2), tokens()
    base = s(t, Y)
    for read in s.reads:
        read.chunk = 1
    assert torch.allclose(base, s(t, Y), atol=1e-5)


def test_the_masked_read_equals_the_gathered_read_in_output_and_gradient():
    """The top-k read as one attention over every source patch with a block mask is the same function."""
    torch.manual_seed(0)
    read = LabelRead(D, heads=2).eval()
    photos, n_sources, k = 80, 64, 32
    t = torch.randn(1, photos, P, D, requires_grad=True)
    sources, labels = t[:, :n_sources], torch.randn(1, n_sources, D)
    nearest = nearest_photos(t, sources, k)
    gathered = read(t, sources, labels, nearest)
    masked = read.forward_masked(t, sources, labels, nearest)
    assert torch.allclose(gathered, masked, atol=1e-5)
    assert neighbour_mask(nearest, n_sources).sum(-1).tolist() == [[k] * photos]
    g1 = torch.autograd.grad(gathered.square().sum(), t)[0]
    g2 = torch.autograd.grad(masked.square().sum(), t)[0]
    assert torch.allclose(g1, g2, atol=1e-4)
