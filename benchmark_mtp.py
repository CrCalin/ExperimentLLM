"""Tiny trainer-style smoke benchmark for baseline and sparse full-width H2."""
import argparse, json, time, torch
from experimentlm import ExperimentConfig, build_model
from experimentlm.optim import build_optimizer, build_schedule, prepare_model_for_optimizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--compile", action="store_true", help="include torch.compile; first-run compilation is excluded only after warmup")
    args = parser.parse_args()
    device = torch.device(args.device)
    for path in ("configs/xielu_test.json", "configs/xielu_mtp.json"):
        c = ExperimentConfig.load(path)
        m = build_model(c.model).to(device).train()
        m = prepare_model_for_optimizer(c.training.optimizer, m)
        optimizer = build_optimizer(c.training.optimizer, m.named_parameters())
        scheduler = build_schedule(c.training.schedule, optimizer, args.warmup + args.steps)
        if args.compile:
            m = torch.compile(m, mode=c.training.compile_mode)
        model_for_metrics = getattr(m, "_orig_mod", m)
        batch, sequence = args.batch_size, args.sequence_length
        x = torch.randint(0, c.model.vocab_size, (batch, sequence), device=device)
        if c.model.num_mtp_heads == 1:
            y = torch.roll(x, -1, 1); y[:, -1] = -100
        else:
            y = torch.stack([torch.roll(x, -h, 1) for h in range(1, c.model.num_mtp_heads + 1)])
            for h in range(1, c.model.num_mtp_heads + 1): y[h - 1, :, -h:] = -100
        def trainer_step():
            m.zero_grad(set_to_none=True)
            for _ in range(args.accumulation):
                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    _, loss = m(x, y, mtp_active=True, mtp_parity=0)
                    (loss / args.accumulation).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), c.training.grad_clip)
            optimizer.step(); scheduler.step()
            return loss
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.warmup): trainer_step()
        if device.type == "cuda": torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(args.steps):
            loss = trainer_step()
        if device.type == "cuda": torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - started) / args.steps
        peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        isolated = None
        if c.model.num_mtp_heads == 2:
            hidden = torch.randn(batch, sequence, c.model.d_model, device=device)
            h2_head = model_for_metrics.mtp_heads[0]
            h2_input = hidden.reshape(-1, c.model.d_model)[::2] if (
                c.model.mtp_head_type == "full_width" and c.model.mtp_position_fraction == 0.5
            ) else hidden
            def h2_forward():
                h2_head.zero_grad(set_to_none=True)
                if c.model.mtp_head_type == "full_width":
                    value = model_for_metrics.lm_head(h2_head(h2_input))
                else:
                    value = h2_head(h2_input)
                value.float().square().mean().backward()
            for _ in range(args.warmup): h2_forward()
            if device.type == "cuda": torch.cuda.synchronize(device)
            isolated_start = time.perf_counter()
            for _ in range(args.steps): h2_forward()
            if device.type == "cuda": torch.cuda.synchronize(device)
            isolated = (time.perf_counter() - isolated_start) / args.steps
        print(json.dumps({"config": path, "parameters": model_for_metrics.num_parameters, "step_seconds": elapsed, "tok_per_s": batch * sequence * args.accumulation / elapsed, "peak_vram_gib": peak, "isolated_h2_head_seconds": isolated, "loss": float(loss.detach()), "horizon_losses": [float(v) for v in model_for_metrics.last_horizon_losses]}))
if __name__ == "__main__": main()
