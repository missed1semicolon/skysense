import os
import io
import base64
import json
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import runpod
from PIL import Image, ImageOps

from transformers import AutoModel


MODEL_ID = os.getenv("MODEL_ID", "BiliSakura/SkySensepp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
HR_SIZE = int(os.getenv("HR_SIZE", "512"))
SPECTRAL_SIZE = int(os.getenv("SPECTRAL_SIZE", "256"))
MAX_OVERLAY_SIZE = int(os.getenv("MAX_OVERLAY_SIZE", "1024"))

model = None


def decode_base64_bytes(value: str) -> bytes:
    if not value:
        raise ValueError("Base64 data is empty.")

    value = value.strip()

    if value.startswith("data:"):
        try:
            value = value.split(",", 1)[1]
        except IndexError as exc:
            raise ValueError("Invalid data URL.") from exc

    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError(f"Invalid Base64 data: {exc}") from exc


def decode_image(value: str) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(decode_base64_bytes(value)))
        image = ImageOps.exif_transpose(image).convert("RGB")
        return image
    except Exception as exc:
        raise ValueError(f"Unable to decode image: {exc}") from exc


def decode_npy(value: str) -> np.ndarray:
    try:
        raw = decode_base64_bytes(value)
        array = np.load(io.BytesIO(raw), allow_pickle=False)
        return np.asarray(array)
    except Exception as exc:
        raise ValueError(
            "Expected a Base64-encoded NumPy .npy array."
        ) from exc


def resize_hr(image: Image.Image) -> torch.Tensor:
    image = image.resize((HR_SIZE, HR_SIZE), Image.Resampling.BICUBIC)
    array = np.asarray(image).astype(np.float32) / 255.0

    # SkySense++ expects CHW tensors. The public HF checkpoint's model
    # interface accepts HR RGB as (B, 3, H, W).
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()

    # The public checkpoint is a remote-sensing foundation model rather
    # than an ImageNet VLM. Keep the raw 0..1 HR representation unless the
    # caller explicitly supplies a different preprocessing contract.
    return tensor


def prepare_spectral(
    array: np.ndarray,
    channels: int,
    name: str,
) -> torch.Tensor:
    array = np.asarray(array, dtype=np.float32)

    # Accepted:
    #   S,C,H,W
    #   C,S,H,W
    #   C,H,W
    #   H,W,C
    if array.ndim == 3:
        if array.shape[0] == channels:
            array = array[:, None, :, :]
        elif array.shape[-1] == channels:
            array = np.transpose(array, (2, 0, 1))[:, None, :, :]
        else:
            raise ValueError(
                f"{name} must have {channels} channels; got shape {array.shape}."
            )
    elif array.ndim == 4:
        if array.shape[1] == channels:
            pass
        elif array.shape[0] == channels:
            array = np.transpose(array, (0, 1, 2, 3))
        elif array.shape[-1] == channels:
            array = np.transpose(array, (3, 0, 1, 2))
        else:
            raise ValueError(
                f"Could not identify {channels} channels in {name} shape "
                f"{array.shape}."
            )
    else:
        raise ValueError(
            f"{name} must be 3D or 4D; received shape {array.shape}."
        )

    # We want C,S,H,W before adding the batch dimension.
    if array.ndim != 4 or array.shape[0] != channels:
        raise ValueError(
            f"{name} could not be normalized to (C,S,H,W); got {array.shape}."
        )

    _, steps, height, width = array.shape

    if height != SPECTRAL_SIZE or width != SPECTRAL_SIZE:
        frames = []
        for step in range(steps):
            frame = torch.from_numpy(array[:, step]).unsqueeze(0)
            frame = torch.nn.functional.interpolate(
                frame,
                size=(SPECTRAL_SIZE, SPECTRAL_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            frames.append(frame.squeeze(0))
        array = torch.stack(frames, dim=1).numpy()

    return torch.from_numpy(array).unsqueeze(0).contiguous()


def load_model() -> None:
    global model

    print("=" * 70)
    print("SNZ SKY SENSE++ WORKER STARTING")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"Dtype: {DTYPE}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the SkySense++ worker.")

    print(f"GPU count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(
            f"GPU {index}: {torch.cuda.get_device_name(index)}"
        )

    print("-" * 70)
    print("Loading SkySense++...")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    model = model.eval().to("cuda")

    print("SkySense++ loaded successfully.")
    print(
        "GPU memory allocated: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    print("=" * 70)


def tensor_summary(tensor: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
    if tensor is None:
        return None

    x = tensor.detach().float()
    return {
        "shape": list(x.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def make_activation_overlay(
    features: torch.Tensor,
    original_size: Tuple[int, int],
) -> str:
    """
    Turn the fused representation into a spatial activation-evidence image.

    This is deliberately described as an activation/evidence overlay rather
    than a semantic segmentation mask. SkySense++'s backbone checkpoint is
    a representation model; without a trained downstream segmentation head,
    the feature magnitude is not a class label.
    """
    x = features.detach().float().cpu()

    if x.ndim == 4:
        # B,C,H,W -> spatial magnitude over channels.
        activation = x[0].pow(2).mean(dim=0).sqrt()
    elif x.ndim == 3:
        # B,N,C -> convert token magnitude to a square map where possible.
        activation = x[0].pow(2).mean(dim=-1).sqrt()
        side = int(round(float(activation.numel()) ** 0.5))
        if side * side != activation.numel():
            activation = activation[: side * side]
        if activation.numel() == 0:
            raise ValueError("SkySense++ returned an empty feature tensor.")
        activation = activation.reshape(side, -1)
    else:
        raise ValueError(
            f"Unsupported features_fusion shape: {tuple(x.shape)}"
        )

    activation = activation.unsqueeze(0).unsqueeze(0)
    activation = torch.nn.functional.interpolate(
        activation,
        size=(original_size[1], original_size[0]),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    lo = torch.quantile(activation, 0.02)
    hi = torch.quantile(activation, 0.98)
    activation = (activation - lo) / (hi - lo + 1e-8)
    activation = activation.clamp(0, 1)

    # Grayscale evidence map; RSCoVLM can consume it as an auxiliary image.
    image_array = (activation.numpy() * 255.0).astype(np.uint8)
    image = Image.fromarray(image_array, mode="L")

    if max(image.size) > MAX_OVERLAY_SIZE:
        scale = MAX_OVERLAY_SIZE / max(image.size)
        image = image.resize(
            (
                max(1, int(image.width * scale)),
                max(1, int(image.height * scale)),
            ),
            Image.Resampling.BILINEAR,
        )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@torch.inference_mode()
def run_inference(job_input: Dict[str, Any]) -> Dict[str, Any]:
    if model is None:
        raise RuntimeError("SkySense++ model has not been loaded.")

    hr_b64 = job_input.get("image_b64") or job_input.get("hr_image_b64")
    s1_b64 = (
        job_input.get("s1_array_b64")
        or job_input.get("s1_npy_b64")
        or job_input.get("sentinel1_b64")
    )
    s2_b64 = (
        job_input.get("s2_array_b64")
        or job_input.get("s2_npy_b64")
        or job_input.get("sentinel2_b64")
    )

    if not hr_b64:
        raise ValueError(
            "SkySense++ requires an HR optical image in image_b64 "
            "or hr_image_b64."
        )

    hr_image = decode_image(hr_b64)
    original_size = hr_image.size
    hr_tensor = resize_hr(hr_image).unsqueeze(0)

    s1_tensor = None
    s2_tensor = None

    if s1_b64:
        s1_tensor = prepare_spectral(
            decode_npy(s1_b64),
            channels=2,
            name="Sentinel-1",
        )

    if s2_b64:
        s2_tensor = prepare_spectral(
            decode_npy(s2_b64),
            channels=10,
            name="Sentinel-2",
        )

    # The public SkySense++ interface uses one boolean flag per modality.
    modalities = torch.tensor(
        [[True, s2_tensor is not None, s1_tensor is not None]],
        dtype=torch.bool,
        device="cuda",
    )

    inputs: Dict[str, Any] = {
        "hr_img": hr_tensor.to("cuda", dtype=DTYPE),
        "modality_flag_hr": modalities[:, :1],
        "modality_flag_s2": modalities[:, 1:2],
        "modality_flag_s1": modalities[:, 2:3],
        "return_features": True,
    }

    if s2_tensor is not None:
        inputs["s2_img"] = s2_tensor.to("cuda", dtype=DTYPE)

    if s1_tensor is not None:
        inputs["s1_img"] = s1_tensor.to("cuda", dtype=DTYPE)

    output = model(**inputs)

    features_fusion = output.get("features_fusion")
    if features_fusion is None:
        raise RuntimeError(
            "SkySense++ did not return features_fusion. "
            "The selected checkpoint/model interface is incompatible "
            "with this worker."
        )

    overlay_b64 = make_activation_overlay(
        features_fusion,
        original_size=original_size,
    )

    feature_shape = list(features_fusion.shape)

    evidence = {
        "feature_representation": "features_fusion",
        "feature_shape": feature_shape,
        "feature_summary": tensor_summary(features_fusion),
        "hr_feature_summary": tensor_summary(
            output.get("features_hr")
            if isinstance(output.get("features_hr"), torch.Tensor)
            else None
        ),
        "s2_feature_summary": tensor_summary(
            output.get("features_s2")
            if isinstance(output.get("features_s2"), torch.Tensor)
            else None
        ),
        "s1_feature_summary": tensor_summary(
            output.get("features_s1")
            if isinstance(output.get("features_s1"), torch.Tensor)
            else None
        ),
        "modalities_present": {
            "hr": True,
            "s2": s2_tensor is not None,
            "s1": s1_tensor is not None,
        },
        "interpretation": (
            "SkySense++ fused representation statistics and a spatial "
            "activation overlay. This is auxiliary model evidence, not "
            "ground-truth segmentation."
        ),
    }

    return {
        "response_text": "",
        "model_used": MODEL_ID,
        "fusion_context": {
            "model": "SkySense++",
            "analysis_mode": "multimodal_feature_extraction",
            "modality_relationship": (
                "HR optical"
                + (" + Sentinel-2" if s2_tensor is not None else "")
                + (" + Sentinel-1 SAR" if s1_tensor is not None else "")
            ),
            "evidence": evidence,
            "evidence_overlay_b64": overlay_b64,
            "overlay_type": "fused_feature_activation",
            "ground_truth": False,
        },
        "metadata": {
            "original_image_size": {
                "width": original_size[0],
                "height": original_size[1],
            },
            "hr_model_size": [HR_SIZE, HR_SIZE],
            "feature_shape": feature_shape,
            "device": DEVICE,
            "dtype": str(DTYPE),
        },
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input", {})

    if not isinstance(job_input, dict):
        return {
            "response_text": "",
            "model_used": MODEL_ID,
            "error": "RunPod job input must be an object/dictionary.",
        }

    try:
        result = run_inference(job_input)

        print("=" * 70)
        print("SKYSENSE++ JOB COMPLETE")
        print(json.dumps({
            "model": result.get("model_used"),
            "analysis_mode": result.get(
                "fusion_context", {}
            ).get("analysis_mode"),
            "modalities": result.get(
                "fusion_context", {}
            ).get("evidence", {}).get("modalities_present"),
            "feature_shape": result.get(
                "fusion_context", {}
            ).get("evidence", {}).get("feature_shape"),
        }, indent=2))
        print("=" * 70)

        return result

    except Exception as exc:
        print("=" * 70)
        print("SKYSENSE++ INFERENCE ERROR")
        print("=" * 70)
        traceback.print_exc()
        print("=" * 70)

        return {
            "response_text": "",
            "model_used": MODEL_ID,
            "fusion_context": None,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    load_model()

    print("Starting RunPod serverless worker...")

    runpod.serverless.start({
        "handler": handler,
    })
