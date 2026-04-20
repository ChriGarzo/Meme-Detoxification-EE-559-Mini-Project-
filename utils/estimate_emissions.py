"""
Estimate CO2 emissions for a completed job when CodeCarbon failed to save.

Reproduces CodeCarbon's estimation methodology using known hardware specs
and a measured runtime. Use this when the EmissionsTracker ran successfully
during the job but failed to persist its CSV (e.g. due to a permission error).

Methodology (mirrors CodeCarbon v2.x):
  energy_kWh = (gpu_W + cpu_W + ram_W) * duration_hours / 1000
  emissions_kg = energy_kWh * carbon_intensity_kg_per_kWh

GPU power: A100-SXM4-40GB TDP is 400W; actual draw during LLaVA batch_size=1
           autoregressive inference is ~35-45% of TDP (~160W average).
CPU power: Cluster CPUs typically ~200W TDP; CodeCarbon uses TDP * utilisation.
           At ~30% CPU utilisation during GPU-bound inference → ~60W.
RAM power: CodeCarbon uses 0.3725 W/GB. Job requested 32 GB → ~12W.
Carbon intensity: EPFL is in Lausanne, Switzerland.
           Swiss grid is dominated by hydro (~60%) + nuclear (~33%).
           Electricitymap.org value for CH: ~23 g CO2/kWh = 0.023 kg/kWh.
           (CodeCarbon uses this value when it can geolocate Switzerland.)

Usage:
    python utils/estimate_emissions.py \\
        --duration_seconds 32161 \\
        --output_path /scratch/hmr_stage1_output/emissions_estimated.csv

    # To use a different carbon intensity (e.g. EU average):
    python utils/estimate_emissions.py \\
        --duration_seconds 32161 \\
        --carbon_intensity_g_per_kwh 276 \\
        --output_path /scratch/hmr_stage1_output/emissions_estimated.csv
"""

import argparse
import csv
import datetime
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware defaults (A100-SXM4-40GB cluster node, EPFL RCP)
# ---------------------------------------------------------------------------

# A100-SXM4-40GB TDP: 400W
# LLaVA-Next 7B 4-bit, batch_size=1, autoregressive decoding → GPU is mostly
# waiting between tokens. Measured as ~35-45% TDP on similar workloads.
GPU_TDP_W: float = 400.0
GPU_UTILISATION: float = 0.40          # conservative estimate

# Typical dual-socket cluster node CPU TDP (e.g. 2× Intel Xeon 5318Y = 2×165W)
# At ~30% utilisation during GPU-bound inference.
CPU_TDP_W: float = 330.0
CPU_UTILISATION: float = 0.30

# RAM: CodeCarbon uses 0.3725 W per GB
RAM_GB: float = 32.0
RAM_W_PER_GB: float = 0.3725

# Switzerland carbon intensity (electricitymap CH zone, 2024 average)
CH_CARBON_INTENSITY_G_PER_KWH: float = 23.0


def estimate_emissions(
    duration_seconds: float,
    gpu_w: float = GPU_TDP_W * GPU_UTILISATION,
    cpu_w: float = CPU_TDP_W * CPU_UTILISATION,
    ram_w: float = RAM_GB * RAM_W_PER_GB,
    carbon_intensity_g_per_kwh: float = CH_CARBON_INTENSITY_G_PER_KWH,
) -> dict:
    """
    Estimate CO2 emissions given hardware power and runtime.

    Returns a dict with energy (kWh), emissions (kg CO2), and emissions (g CO2).
    """
    duration_hours = duration_seconds / 3600.0
    total_power_w = gpu_w + cpu_w + ram_w
    energy_kwh = total_power_w * duration_hours / 1000.0
    emissions_kg = energy_kwh * (carbon_intensity_g_per_kwh / 1000.0)
    emissions_g = emissions_kg * 1000.0

    return {
        "duration_seconds": duration_seconds,
        "duration_hours": duration_hours,
        "gpu_power_w": gpu_w,
        "cpu_power_w": cpu_w,
        "ram_power_w": ram_w,
        "total_power_w": total_power_w,
        "energy_kwh": energy_kwh,
        "carbon_intensity_g_per_kwh": carbon_intensity_g_per_kwh,
        "emissions_kg": emissions_kg,
        "emissions_g": emissions_g,
    }


def write_csv(result: dict, output_path: str, job_name: str = "hmr-stage1") -> None:
    """Write estimated emissions to a CSV matching CodeCarbon's output format."""
    row = {
        "timestamp": datetime.datetime.now().isoformat(),
        "project_name": job_name,
        "run_id": "estimated",
        "experiment_id": "estimated",
        "duration": result["duration_seconds"],
        "emissions": result["emissions_kg"],
        "emissions_rate": result["emissions_kg"] / result["duration_hours"],
        "cpu_power": result["cpu_power_w"],
        "gpu_power": result["gpu_power_w"],
        "ram_power": result["ram_power_w"],
        "cpu_energy": result["cpu_power_w"] * result["duration_hours"] / 1000.0,
        "gpu_energy": result["gpu_power_w"] * result["duration_hours"] / 1000.0,
        "ram_energy": result["ram_power_w"] * result["duration_hours"] / 1000.0,
        "energy_consumed": result["energy_kwh"],
        "country_name": "Switzerland",
        "country_iso_code": "CHE",
        "region": "Lausanne",
        "cloud_provider": "on_premise",
        "cloud_region": "epfl_rcp",
        "os": "Linux",
        "python_version": "3.12",
        "codecarbon_version": "estimated",
        "cpu_count": 4,
        "cpu_model": "Intel Xeon (estimated)",
        "gpu_count": 1,
        "gpu_model": "NVIDIA A100-SXM4-40GB",
        "longitude": 6.5668,
        "latitude": 46.5197,
        "ram_total_size": RAM_GB,
        "tracking_mode": "estimated",
        "on_cloud": "N",
    }

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    logger.info(f"Estimated emissions saved to: {output_path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Estimate CO2 emissions for a completed job")
    parser.add_argument(
        "--duration_seconds", type=float, required=True,
        help="Measured wall-clock runtime in seconds (read from job logs)"
    )
    parser.add_argument(
        "--carbon_intensity_g_per_kwh", type=float,
        default=CH_CARBON_INTENSITY_G_PER_KWH,
        help=f"Grid carbon intensity in g CO2/kWh (default: {CH_CARBON_INTENSITY_G_PER_KWH} for Switzerland)"
    )
    parser.add_argument(
        "--gpu_utilisation", type=float, default=GPU_UTILISATION,
        help=f"GPU utilisation fraction 0-1 (default: {GPU_UTILISATION})"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to save the estimated emissions CSV"
    )
    parser.add_argument(
        "--job_name", type=str, default="hmr-stage1",
        help="Job name to record in the CSV"
    )
    args = parser.parse_args()

    gpu_w = GPU_TDP_W * args.gpu_utilisation
    cpu_w = CPU_TDP_W * CPU_UTILISATION
    ram_w = RAM_GB * RAM_W_PER_GB

    result = estimate_emissions(
        duration_seconds=args.duration_seconds,
        gpu_w=gpu_w,
        cpu_w=cpu_w,
        ram_w=ram_w,
        carbon_intensity_g_per_kwh=args.carbon_intensity_g_per_kwh,
    )

    print("\n" + "=" * 60)
    print("  Stage 1 CO2 Emission Estimate")
    print("=" * 60)
    print(f"  Runtime:           {result['duration_seconds']:.0f}s  ({result['duration_hours']:.2f}h)")
    print(f"  GPU power (est.):  {result['gpu_power_w']:.1f}W  (A100 TDP×{args.gpu_utilisation:.0%})")
    print(f"  CPU power (est.):  {result['cpu_power_w']:.1f}W  (TDP×{CPU_UTILISATION:.0%})")
    print(f"  RAM power (est.):  {result['ram_power_w']:.1f}W  ({RAM_GB:.0f} GB × {RAM_W_PER_GB} W/GB)")
    print(f"  Total power:       {result['total_power_w']:.1f}W")
    print(f"  Energy consumed:   {result['energy_kwh']:.4f} kWh")
    print(f"  Carbon intensity:  {result['carbon_intensity_g_per_kwh']:.1f} g CO2/kWh (Switzerland)")
    print(f"  CO2 emissions:     {result['emissions_kg']:.6f} kg  ({result['emissions_g']:.4f} g)")
    print("=" * 60)
    print("  Note: GPU utilisation is estimated (~35-45% for LLaVA")
    print("  batch_size=1 autoregressive inference). The true value")
    print("  depends on actual NVML power readings that were not saved.")
    print("=" * 60 + "\n")

    write_csv(result, args.output_path, job_name=args.job_name)


if __name__ == "__main__":
    main()
