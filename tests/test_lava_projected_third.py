import torch
from omegaconf import OmegaConf
from pathlib import Path

from models.model_runner import ModelFactory
from models.vla_model_fm import projected_logsignature_level_three
from test_lava_episode_balanced import balanced_inputs


def test_projected_third_matches_bch_for_two_segments_and_backpropagates():
    projection = torch.eye(2)
    path = torch.tensor([[1., 0., 0.], [0., 1., 0.]], requires_grad=True)
    actual = projected_logsignature_level_three(path, projection).reshape(3, 3, 3)

    def bracket(a, b):
        return torch.einsum("i,j->ij", a, b) - torch.einsum("i,j->ij", b, a)

    x, y = path[0], path[1]
    xy = bracket(x, y)
    expected = (torch.einsum("i,jk->ijk", x, xy)
                - torch.einsum("ij,k->ijk", xy, x)
                - torch.einsum("i,jk->ijk", y, xy)
                + torch.einsum("ij,k->ijk", xy, y)) / 12
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert projected_logsignature_level_three(path[:1], projection).count_nonzero() == 0
    actual.square().sum().backward()
    assert torch.isfinite(path.grad).all() and path.grad.norm() > 0


def test_projected_third_full_objective_and_monitoring():
    config = OmegaConf.load(Path(__file__).parents[1] / 'configs/robotwin_lava_v66_projected3_action8.yaml')
    config.model.action_expert.hidden_size = 32
    config.model.action_expert.depth = 8
    config.model.action_expert.num_heads = 4
    config.model.future_feat.enabled = False
    model = ModelFactory.create_action_model(config, 16, 1).bfloat16()
    kw = balanced_inputs((1, 2, 4), (1, 1, 1))
    kw['action_hidden'] = kw['action_hidden'].detach().bfloat16().requires_grad_()
    kw['lava_context'] = kw['lava_context'].bfloat16()
    loss, info = model.compute_lava_loss(**kw)
    assert torch.isfinite(loss)
    assert model.lava_third_projection.shape == (128, 31)
    assert 'third_candidate_acc_gain' in info
    assert info['third_action_raw_norm'] > 0 and info['third_world_raw_norm'] > 0
    assert 0 <= info['third_action_energy_fraction'] <= 1
    loss.backward()
    assert torch.isfinite(kw['action_hidden'].grad).all()
