# RGB, depth and NIR inputs

These corresponding inputs are from the `微弱光` lighting folder, image `1.png`.

![RGB, depth and NIR](images/rgb_depth_nir.png)

| Modality | Original | Display |
| --- | --- | --- |
| RGB | [8-bit RGB](inputs/rgb.png) | [RGB](images/rgb.png) |
| Depth | [16-bit PNG](inputs/depth.png) | [Depth](images/depth_display.png) |
| NIR | [16-bit PNG](inputs/nir.png) | [NIR](images/nir_display.png) |

## Preview

```bash
uv sync --locked
uv run --locked python -m tefed preview --example examples/tomato_trimodal --output runs/tomato_preview
```

The command preserves original inputs and writes the display images, combined preview and `preview.json`. The fixed limits in [manifest.json](manifest.json) use positive-pixel 1st/99th percentiles. Depth uses the [viridis palette](display_palette.json), NIR uses grayscale, and zero values appear black.

Use `inputs/` for data processing. For predictions, follow [single-sample inference](../../docs/single_sample_inference.md) with the checkpoint, calibration and training preprocessing.

The supplied [class_names.json](class_names.json) uses `0=non-ripe`, `1=ripe`. Use the order stored in the trained checkpoint for detection.

Data: Zhang et al., *A dataset of multimodal images of tomato fruits at different stages of maturity*, [China Scientific Data (2025)](https://doi.org/10.11922/11-6035.csd.2023.0154.zh).
