import functools
from typing import Optional

import numpy as np
import torch

from flower_vla.dataset.oxe.transforms import generate_policy_prompt, get_action_space_index
from flower_vla.eval.simpler.flower_inference_wrapper import UhaInference as SimplerUhaInference


# Normalization stats (p99/p01) for the first 6 action dims per LIBERO benchmark.
# Extracted from conf/datamodule/data_statistics/*.yaml
LIBERO_NORM_STATS = {
    "libero_10": {
        "p99": [0.7714285850524902, 0.8464285731315613, 0.9375, 0.13928571343421936, 0.15964286029338837, 0.3246428668498993],
        "p01": [-0.6348214149475098, -0.7741071581840515, -0.7633928656578064, -0.09749999642372131, -0.14819999992847435, -0.2742857038974762],
    },
    "libero_goal": {
        "p99": [0.9375, 0.9107142686843872, 0.9375, 0.20357142388820648, 0.26357144117355347, 0.375],
        "p01": [-0.8785714507102966, -0.7553571462631226, -0.9375, -0.1510714292526245, -0.1639285683631897, -0.13777500048279764],
    },
    # spatial/object share libero_10 stats as fallback (no separate yaml)
    "libero_spatial": {
        "p99": [0.7714285850524902, 0.8464285731315613, 0.9375, 0.13928571343421936, 0.15964286029338837, 0.3246428668498993],
        "p01": [-0.6348214149475098, -0.7741071581840515, -0.7633928656578064, -0.09749999642372131, -0.14819999992847435, -0.2742857038974762],
    },
    "libero_object": {
        "p99": [0.7714285850524902, 0.8464285731315613, 0.9375, 0.13928571343421936, 0.15964286029338837, 0.3246428668498993],
        "p01": [-0.6348214149475098, -0.7741071581840515, -0.7633928656578064, -0.09749999642372131, -0.14819999992847435, -0.2742857038974762],
    },
}


class LiberoInference(SimplerUhaInference):
    """Inference wrapper for LIBERO benchmark evaluation.

    Extends UhaInference with LIBERO-specific action space, prompt format,
    and denormalization stats.
    """

    def __init__(
        self,
        saved_model_base_dir: str,
        saved_model_path: str,
        benchmark_name: str = "libero_10",
        pred_action_horizon: int = 10,
        multistep: int = 5,
        num_sampling_steps: int = 1,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_ema: bool = False,
        use_torch_compile: bool = False,
        ensemble_strategy: str = "false",
        cfg_lambda: float = 1.0,
    ):
        super().__init__(
            saved_model_base_dir=saved_model_base_dir,
            saved_model_path=saved_model_path,
            image_size=224,
            pred_action_horizon=pred_action_horizon,
            action_scale=1.0,
            policy_setup="skip_policy_setup",
            device=device,
            ensemble_strategy=ensemble_strategy,
            multistep=multistep,
            num_sampling_steps=num_sampling_steps,
            use_ema=use_ema,
            use_torch_compile=use_torch_compile,
            cfg_lambda=cfg_lambda,
        )

        # LIBERO uses EEF_POS single-arm, no proprio (ZeroEncoder)
        self.agent.agent.use_proprio = False

        # Prompt format matching training transform (transforms.py:1749)
        self.format_instruction = functools.partial(
            generate_policy_prompt,
            robot_name="Franka Panda",
            action_space="delta end-effector",
            num_arms="1",
            prompt_style="minimal",
        )

        # Action space = EEF_POS, 1-arm, velocity (transforms.py:1760)
        self.action_space_index = torch.tensor([
            get_action_space_index('EEF_POS', 1, 'velocity', return_tensor=False)
        ])
        # All LIBERO datasets have frequency=10
        self.frequency = torch.tensor([10])

        # Denormalization stats (first 6 dims only, gripper handled separately)
        stats = LIBERO_NORM_STATS.get(benchmark_name, LIBERO_NORM_STATS["libero_10"])
        self.max_values = torch.tensor(stats["p99"])
        self.min_values = torch.tensor(stats["p01"])

    def step(self, obs_dict: dict, task_description: Optional[str] = None) -> np.ndarray:
        """Run one inference step for LIBERO.

        Args:
            obs_dict: LIBERO environment observation dict (must contain 'agentview_image').
            task_description: Natural language task instruction.

        Returns:
            7D numpy action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
            where gripper is in LIBERO convention (-1=open, 1=close).
        """
        task_description = self.format_instruction(task_description)
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)
                self.agent.agent.reset()
                self.act_chunk_deque.clear()

        # Extract and preprocess image
        image = obs_dict["agentview_image"]  # (H, W, 3) uint8
        assert image.dtype == np.uint8
        image = self._resize_image(image)  # resize to 224x224
        image = torch.from_numpy(np.moveaxis(image, -1, 0)).unsqueeze(0).unsqueeze(0).to(device=self.device)
        # image shape: [1, 1, 3, 224, 224]

        input_observation = {
            "observation": {
                "image_primary": image,
                "pad_mask_dict": {"image_primary": torch.ones(1, 1).bool().to(device=self.device)},
            },
            "task": {
                "language_instruction": self.task_description_embedding,
                "frequency": self.frequency,
                "action_space_index": self.action_space_index,
            }
        }

        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                unscaled_raw_actions = self.agent(input_observation).cpu()

        # Trim to action dim (7 for EEF_POS single arm)
        unscaled_raw_actions = unscaled_raw_actions[:self.action_index.get_action_dim(self.action_space_index)]

        # Denormalize: first 6 dims via rescale_to_range, keep gripper separate
        eef_actions = self.rescale_to_range(unscaled_raw_actions[..., :6])
        gripper_model = unscaled_raw_actions[..., 6:7]  # model space: 1=open, 0=close

        # Convert gripper: model (1=open, 0=close) -> LIBERO (-1=open, 1=close)
        gripper_env = 1.0 - 2.0 * gripper_model

        raw_actions = torch.cat([eef_actions, gripper_env], dim=-1).detach()
        assert raw_actions.shape[-1] == 7, f"Expected 7D action, got {raw_actions.shape}"

        return raw_actions.numpy()
