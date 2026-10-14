import unittest, torch
from experimentlm import ExperimentConfig, build_model
from experimentlm.data import TokenStream
import tempfile, numpy as np

class MTPTests(unittest.TestCase):
    def model(self, n=2, tied=False, mtp_loss_weight=0.1, head_type="full_width", position_fraction=0.5):
        c = ExperimentConfig.load("configs/xielu_test.json")
        raw = c.to_dict(); raw["model"].update(vocab_size=32, context_length=8, d_model=32, tie_embeddings=tied, num_mtp_heads=n, mtp_loss_weight=mtp_loss_weight, mtp_head_type=head_type, mtp_position_fraction=position_fraction)
        for layer in raw["model"]["layers"]:
            layer["args"]["attention"]["args"].update(n_heads=4, head_dim=8)
            layer["args"]["mlp"]["args"]["d_ff"] = 64
        return build_model(ExperimentConfig.from_dict(raw).model)
    def test_h2_head_shapes_and_gradients(self):
        m = self.model(); x=torch.randint(0,32,(2,8)); y=torch.randint(0,32,(2,2,8)); m(x,y)[1].backward()
        head = m.mtp_heads[0]
        self.assertEqual(head[0].normalized_shape, (32,))
        self.assertEqual(head[1].weight.shape, (32, 32))
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))
    def test_no_future_leakage(self):
        m=self.model(2); x=torch.randint(0,32,(1,8)); a=m(x)[0]; x[:, -1]=0; b=m(x)[0]; self.assertTrue(torch.allclose(a[:,:-1],b[:,:-1]))
    def test_standard_disabled_and_tied(self):
        m=self.model(1, True); x=torch.randint(0,32,(1,8)); logits,_=m(x, torch.randint(0,32,(1,8))); self.assertTrue(torch.equal(m.lm_head.weight,m.wte.weight)); self.assertEqual(logits.shape,(1,8,32))
    def test_mtp_model_accepts_standard_evaluation_labels(self):
        m=self.model(); x=torch.randint(0,32,(1,8)); _, loss=m(x, torch.randint(0,32,(1,8))); self.assertTrue(torch.isfinite(loss)); self.assertEqual(len(m.last_horizon_losses), 1)
    def test_auxiliary_loss_weighting(self):
        m=self.model(); x=torch.randint(0,32,(1,8)); y=torch.randint(0,32,(2,1,8))
        _, loss=m(x,y); horizon_losses=m.last_horizon_losses
        self.assertTrue(torch.allclose(loss.detach(), horizon_losses[0] + 0.1 * horizon_losses[1]))
        m=self.model(mtp_loss_weight=0.0); _, zero_loss=m(x,y)
        self.assertTrue(torch.equal(zero_loss.detach(), m.last_horizon_losses[0]))
    def test_auxiliary_gradients_follow_weight(self):
        x=torch.randint(0,32,(1,8)); y=torch.randint(0,32,(2,1,8))
        m=self.model(); m(x,y)[1].backward(); self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.mtp_heads.parameters()))
        m=self.model(mtp_loss_weight=0.0); m(x,y)[1].backward(); self.assertTrue(all(p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad)) for p in m.mtp_heads.parameters()))
    def test_stream_masks_horizons(self):
        with tempfile.NamedTemporaryFile() as f:
            np.arange(9,dtype=np.uint16).tofile(f.name); _, y=next(iter(TokenStream(f.name, 8, 1, prediction_horizons=2)))
            self.assertEqual(y.shape,(2,1,8)); self.assertEqual(y[0,0,-1], 8); self.assertEqual(y[1,0,-1], -100)
    def test_h2_selects_deterministic_half_of_valid_positions(self):
        m=self.model(); x=torch.randint(0,32,(2,8)); y=torch.randint(0,32,(2,2,8)); y[1,0,-2:] = -100; y[1,1,-1:] = -100
        m(x,y)
        valid = y[1].reshape(-1).ne(-100).nonzero(as_tuple=False).flatten()
        expected = valid[::2]
        self.assertEqual(m.last_mtp_positions_total, valid.numel())
        self.assertEqual(m.last_mtp_positions_used, expected.numel())
        self.assertTrue(torch.equal(m.last_mtp_selected_indices, expected))
        m(x,y)
        self.assertTrue(torch.equal(m.last_mtp_selected_indices, expected))
    def test_h2_logits_only_use_selected_rows(self):
        m=self.model(); x=torch.randint(0,32,(2,8)); y=torch.randint(0,32,(2,2,8)); captured=[]
        hook=m.lm_head.register_forward_hook(lambda _module, args, _output: captured.append(args[0].shape[0]))
        m(x,y); hook.remove()
        self.assertEqual(captured, [2, m.last_mtp_positions_used])
    def test_alternating_parity_covers_both_position_sets(self):
        m=self.model(); x=torch.randint(0,32,(1,8)); y=torch.randint(0,32,(2,1,8)); y[1,0,-2:] = -100
        m(x,y,mtp_step=0); even = m.last_mtp_selected_indices.clone()
        m(x,y,mtp_step=1); odd = m.last_mtp_selected_indices.clone()
        self.assertTrue(torch.equal(even, torch.tensor([0, 2, 4])))
        self.assertTrue(torch.equal(odd, torch.tensor([1, 3, 5])))
        self.assertEqual(set(even.tolist()) | set(odd.tolist()), set(range(6)))
    def test_mtp_cutoff_is_at_step_1000_and_skips_auxiliary_path(self):
        m=self.model(); x=torch.randint(0,32,(1,8)); y=torch.randint(0,32,(2,1,8)); calls=[]
        hook=m.mtp_heads[0].register_forward_hook(lambda *_args: calls.append(True))
        m(x,y,mtp_step=999)
        self.assertEqual(len(calls), 1); self.assertEqual(len(m.last_horizon_losses), 2)
        calls.clear(); m.zero_grad(set_to_none=True); _, cutoff_loss=m(x,y,mtp_step=1000); cutoff_loss.backward()
        hook.remove()
        self.assertEqual(calls, [])
        self.assertEqual(len(m.last_horizon_losses), 1)
        self.assertTrue(all(parameter.grad is None for parameter in m.mtp_heads.parameters()))
    def test_cutoff_matches_plain_ntp_forward_and_loss(self):
        torch.manual_seed(123)
        mtp=self.model(); torch.manual_seed(123); baseline=self.model(n=1)
        x=torch.randint(0,32,(1,8)); y=torch.randint(0,32,(2,1,8))
        mtp_logits, mtp_loss=mtp(x,y,mtp_step=1000); baseline_logits, baseline_loss=baseline(x,y[0])
        self.assertTrue(torch.equal(mtp_logits, baseline_logits))
        self.assertTrue(torch.equal(mtp_loss, baseline_loss))
    def test_h2_config_has_only_one_auxiliary_head(self):
        config = ExperimentConfig.load("configs/xielu_mtp.json")
        model = build_model(config.model)
        self.assertEqual(config.model.num_mtp_heads, 2)
        self.assertEqual(config.model.mtp_position_fraction, 0.5)
        self.assertEqual(config.model.mtp_active_steps, 1000)
        self.assertTrue(config.model.mtp_alternating_parity)
        self.assertEqual(config.model.mtp_head_type, "full_width")
        self.assertEqual(len(model.mtp_heads), 1)
        self.assertFalse(any(name.startswith("mtp_heads.1") for name, _ in model.named_parameters()))
    def test_disabling_mtp_recovers_standard_behavior(self):
        torch.manual_seed(123)
        baseline = build_model(ExperimentConfig.load("configs/xielu_test.json").model)
        torch.manual_seed(123)
        h2 = build_model(ExperimentConfig.load("configs/xielu_mtp.json").model)
        x = torch.randint(0, 8192, (1, 8))
        self.assertTrue(torch.equal(baseline(x)[0], h2(x)[0]))
if __name__ == "__main__": unittest.main()
