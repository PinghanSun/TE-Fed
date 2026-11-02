# Material-based infrared preview

`render_scene_ir.py` computes IR from scene geometry, material spectra, emissivity and temperature.

Two example presets are available:

| Preset | Band | Image formation |
| --- | --- | --- |
| `nir` | 0.75–1.05 µm | Lambertian reflection of direct and sky irradiance, with direct shadows |
| `lwir` | 8–14 µm | Planck emission plus reflected environmental radiation |

The presets provide example material, temperature and sensor parameters. Use `--config` to supply calibration for a specific sensor.

Run from the source repository, using your local scene path:

```bash
blender --background path/to/vehicle_environment.blend --python scripts/render_scene_ir.py -- --preset lwir --width 640 --samples 4 --with-rgb --output runs/vehicle_lwir
```

Use `--preset nir` for reflection and `--camera Environment_Street` for the street view. `--with-rgb` adds a Cycles image at the same camera, size and geometry. RGB uses scene lights; IR uses the spectral configuration. Choose a new output directory.

Outputs include `radiance.npy` (response-weighted W m⁻² sr⁻¹), 16-bit sensor codes in `ir.png` and `ir_dn.npy`, a fixed-window `ir_preview.png`, surface/object index maps, and `depth_m.npy`. Depth is range along the camera ray, rather than camera-axis Z; pixels without a hit are NaN. `ir.json` records the full parameters, surface assignments, source/script hashes and camera. Raw radiance is preserved before gain, noise, clipping or display mapping.

Both images use the same full frame with square pixels. Saved render-border and crop settings are disabled in memory and recorded in `frame_settings`; the source blend is unchanged. Near and far clipping use camera-axis planes.

## Material and calibration interface

Set `tefed_ir_material` on a Blender material or object to one of `rubber`, `metal`, `painted_metal`, `glass`, `vegetation`, `asphalt`, `concrete`, `soil` or `generic`. Optional properties `ir_reflectance`, `ir_emissivity` and `ir_temperature_k` override the profile. Object properties take precedence over material properties. Unassigned materials use explicit name tokens; unmatched names use `generic` and remain visible in the output assignments.

Export a configuration with Python and NumPy:

```bash
python scripts/render_scene_ir.py --preset lwir --write-preset lwir_parameters.json
python scripts/render_scene_ir.py --preset lwir --config lwir_parameters.json --check
```

Or use Blender's bundled Python:

```text
blender --background --factory-startup --python scripts/render_scene_ir.py -- --preset lwir --write-preset lwir_parameters.json
```

The standalone script's `--check` validates configuration. `run_ir_example.py --check` also loads the requested scene and validates its camera and mode. Invalid configurations and mismatched modes return a nonzero exit code.

Pass the same `--config` option to Blender. The JSON contains sensor wavelengths in µm, dimensionless relative energy response, gain/offset/noise, atmospheric transmission, irradiance, temperatures and material profiles. A detector's quantum-efficiency curve must first be converted to an energy response. Scalar reflectance/emissivity or arrays on the sensor grid are accepted. For measured spectra on a separate grid, use `{"wavelength_um": [8, 10, 14], "values": [0.91, 0.93, 0.90]}`; the curve must cover the sensor band. These numbers are syntax examples.

`materials` and `objects` map exact names to overrides, for example `{"objects": {"car_01_hood": {"temperature_k": 310}}}`. JSON overrides take precedence over Blender properties, with object overrides last. Metadata records the supplied overrides and preset values. A calibration file can be linked with `"calibration_reference": {"path": "calibration.json", "sha256": "<64 hex characters>"}`; relative paths resolve beside the configuration, and its content hash is checked.

The model uses opaque, diffuse surfaces and assigned temperatures, without transmission, directional BRDFs, heat transfer or interreflection. Convert collection instances to meshes before rendering. For NIR, the supplied irradiance spectrum is split between a directional source and isotropic sky. For LWIR, opaque reflectance is `1 − emissivity`. Both modes integrate spectral radiance against the supplied sensor response and atmospheric transmission.

The equations follow [Planck's law](https://cie.co.at/eilvterm/17-24-005) and [Lambertian reflection](https://pbr-book.org/3ed-2018/Reflection_Models/Lambertian_Reflection). Measured material spectra are available from the [USGS spectral library](https://www.usgs.gov/labs/spectroscopy-lab/usgs-spectral-library).
