"""LIBERO benchmark rollout evaluation for FLOWER VLA policies.

Usage:
    python -m flower_vla.eval.libero.libero_eval \
        --checkpoint_dir /path/to/run \
        --checkpoint_name checkpoint_20000 \
        --benchmark_name libero_10 \
        --n_eval 20
"""

import argparse
import gc
import os
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import wandb

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from flower_vla.eval.libero.inference_wrapper import LiberoInference
from flower_vla.eval.utils.utils import set_seed


class EvaluateLibero:
    def __init__(
        self,
        model: LiberoInference,
        benchmark_name: str = "libero_10",
        n_eval: int = 20,
        max_steps: int = 520,
        num_videos: int = 3,
        log_dir: str = "results/libero",
    ):
        self.model = model
        self.n_eval = n_eval
        self.max_steps = max_steps
        self.num_videos = num_videos
        self.log_dir = log_dir

        os.makedirs(log_dir, exist_ok=True)

        # Setup LIBERO benchmark
        self.benchmark_name = benchmark_name
        self.benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
        self.task_names = self.benchmark_instance.get_task_names()
        self.num_tasks = self.benchmark_instance.get_num_tasks()

        print(f"Benchmark: {benchmark_name}")
        print(f"Number of tasks: {self.num_tasks}")
        print(f"Evaluations per task: {n_eval}")

    def evaluate_policy(self) -> list[float]:
        """Evaluate all tasks, return per-task success rates."""
        per_task_success = []

        for task_idx in range(self.num_tasks):
            task_name = self.task_names[task_idx]
            print(f"\n[{task_idx+1}/{self.num_tasks}] Evaluating: {task_name}")

            success_rate = self.evaluate_task(task_idx)
            per_task_success.append(success_rate)

            print(f"  Success rate: {success_rate:.1%} ({int(success_rate * self.n_eval)}/{self.n_eval})")

            if wandb.run is not None:
                wandb.log({
                    f"task/{task_name}": success_rate,
                    "progress/tasks_completed": task_idx + 1,
                    "progress/running_mean": np.mean(per_task_success),
                })

        avg_success = np.mean(per_task_success)
        print(f"\nOverall success rate: {avg_success:.1%}")

        if wandb.run is not None:
            wandb.log({
                "final/avg_success_rate": avg_success,
                "final/std_success_rate": np.std(per_task_success),
            })
            # Log table with per-task results
            table = wandb.Table(
                data=[[name, rate] for name, rate in zip(self.task_names, per_task_success)],
                columns=["Task", "Success Rate"],
            )
            wandb.log({"final/results_table": table})

        return per_task_success

    def evaluate_task(self, task_idx: int) -> float:
        """Run n_eval rollouts for a single task, return success rate."""
        task = self.benchmark_instance.get_task(task_idx)
        task_name = self.task_names[task_idx]
        task_description = task.language

        bddl_folder = get_libero_path("bddl_files")
        task_bddl_file = os.path.join(bddl_folder, task.problem_folder, task.bddl_file)

        initial_states = self.benchmark_instance.get_task_init_states(task_idx)

        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=256,
            camera_widths=256,
        )

        num_success = 0
        video_dir = os.path.join(self.log_dir, "videos", task_name.replace(" ", "_"))
        os.makedirs(video_dir, exist_ok=True)

        for ep_idx in range(self.n_eval):
            init_state = initial_states[ep_idx] if ep_idx < len(initial_states) else initial_states[ep_idx % len(initial_states)]
            success, frames = self._run_episode(
                env, init_state, task_description,
                record_video=(ep_idx < self.num_videos),
            )
            num_success += int(success)

            # Save video
            if ep_idx < self.num_videos and frames:
                video_path = os.path.join(video_dir, f"ep{ep_idx}_{'ok' if success else 'fail'}.mp4")
                self._save_video(frames, video_path)
                if wandb.run is not None:
                    wandb.log({
                        f"video/{task_name}/ep{ep_idx}": wandb.Video(video_path, fps=10, format="mp4"),
                    })

        env.close()
        gc.collect()
        return num_success / self.n_eval

    def _run_episode(
        self,
        env,
        init_state: np.ndarray,
        task_description: str,
        record_video: bool = False,
    ) -> tuple[bool, list[np.ndarray]]:
        """Run a single rollout episode."""
        env.reset()
        obs = env.set_init_state(init_state)

        # Dummy steps for physics stabilization
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros(7))

        self.model.reset(self.model.format_instruction(task_description))
        self.model.agent.agent.reset()
        self.model.act_chunk_deque.clear()

        frames = []
        done = False

        for step_idx in range(self.max_steps):
            action = self.model.step(obs, task_description)
            obs, reward, done, info = env.step(action)

            if record_video:
                frame = obs["agentview_image"]
                # Ensure uint8
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                frames.append(frame)

            if done:
                break

        return done, frames

    @staticmethod
    def _save_video(frames: list[np.ndarray], path: str, fps: int = 10):
        """Save frames as an MP4 video."""
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for frame in frames:
            # Convert RGB -> BGR for OpenCV
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()


def main():
    parser = argparse.ArgumentParser(description="LIBERO rollout evaluation")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to run directory containing .hydra/")
    parser.add_argument("--checkpoint_name", required=True, help="Checkpoint folder name (e.g. checkpoint_20000)")
    parser.add_argument("--benchmark_name", default="libero_10", choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"])
    parser.add_argument("--n_eval", type=int, default=20, help="Number of rollouts per task")
    parser.add_argument("--max_steps", type=int, default=520, help="Max env steps per episode")
    parser.add_argument("--num_videos", type=int, default=3, help="Number of rollouts to record per task")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--pred_action_horizon", type=int, default=10)
    parser.add_argument("--multistep", type=int, default=5)
    parser.add_argument("--num_sampling_steps", type=int, default=1)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--use_torch_compile", action="store_true")
    parser.add_argument("--ensemble_strategy", default="false", choices=["false", "act", "cogact"])
    parser.add_argument("--cfg_lambda", type=float, default=1.0)
    parser.add_argument("--wandb_project", default="flower_libero_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", default="results/libero")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init wandb
    run_name = f"{args.benchmark_name}_{args.checkpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
    )

    # Build model
    model = LiberoInference(
        saved_model_base_dir=args.checkpoint_dir,
        saved_model_path=args.checkpoint_name,
        benchmark_name=args.benchmark_name,
        pred_action_horizon=args.pred_action_horizon,
        multistep=args.multistep,
        num_sampling_steps=args.num_sampling_steps,
        device=device,
        use_ema=args.use_ema,
        use_torch_compile=args.use_torch_compile,
        ensemble_strategy=args.ensemble_strategy,
        cfg_lambda=args.cfg_lambda,
    )

    # Run evaluation
    evaluator = EvaluateLibero(
        model=model,
        benchmark_name=args.benchmark_name,
        n_eval=args.n_eval,
        max_steps=args.max_steps,
        num_videos=args.num_videos,
        log_dir=args.log_dir,
    )

    t0 = time.time()
    per_task_success = evaluator.evaluate_policy()
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Per-task: {[f'{s:.1%}' for s in per_task_success]}")
    print(f"Average: {np.mean(per_task_success):.1%}")

    wandb.finish()


if __name__ == "__main__":
    main()
