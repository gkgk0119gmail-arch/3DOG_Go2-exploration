"""Author a Velodyne VLP-16 RTX lidar profile (OmniLidar prim) from the VLP-16 user manual.

Isaac Sim 6.0.1 ships no VLP-16 profile (only Ouster, VLS-128, Hesai XT32, SICK, NVIDIA examples), so this writes
one with the generic rotary lidar schema (template: Isaac/Sensors/NVIDIA/Example_Rotary.usda).

VLP-16 (user manual rev. E):
  * 16 lasers, firing order (laser id 0..15) elevations: -15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15 deg
  * one laser every 2.304 us, a 16-laser firing sequence + recharge every 55.296 us (18.08 kHz)
  * 10 Hz -> 0.199 deg azimuth resolution, range 100 m (we keep the driver's 0.9 m minimum), accuracy +-3 cm
  * rotation clockwise seen from above, single (strongest) return

Usage:
    python make_vlp16_rtx.py out.usda
"""

import argparse

from pxr import Sdf, Usd, UsdGeom, Vt

ELEVATIONS = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
FIRE_DT_NS = 2304
SEQUENCE_NS = 55296


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    args = ap.parse_args()

    stage = Usd.Stage.CreateNew(args.output)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = stage.DefinePrim("/VLP16", "OmniLidar")
    stage.SetDefaultPrim(prim)
    prim.AddAppliedSchema("OmniSensorGenericLidarCoreAPI")

    n = len(ELEVATIONS)
    attrs = {
        "omni:sensor:Core:emitterState:s001:azimuthDeg": (Sdf.ValueTypeNames.FloatArray, [0.0] * n),
        "omni:sensor:Core:emitterState:s001:elevationDeg": (Sdf.ValueTypeNames.FloatArray, [float(e) for e in ELEVATIONS]),
        "omni:sensor:Core:emitterState:s001:channelId": (Sdf.ValueTypeNames.UIntArray, list(range(1, n + 1))),
        "omni:sensor:Core:emitterState:s001:fireTimeNs": (Sdf.ValueTypeNames.UIntArray, [i * FIRE_DT_NS for i in range(n)]),
        "omni:sensor:Core:emitterState:s001:distanceCorrectionM": (Sdf.ValueTypeNames.FloatArray, [0.0] * n),
        "omni:sensor:Core:emitterState:s001:emitterPeakPowerW": (Sdf.ValueTypeNames.FloatArray, [0.0] * n),
        "omni:sensor:Core:emitterState:s001:focalDistM": (Sdf.ValueTypeNames.FloatArray, [0.0] * n),
        "omni:sensor:Core:emitterState:s001:focalSlope": (Sdf.ValueTypeNames.FloatArray, [0.0] * n),
        "omni:sensor:Core:emitterState:s001:horOffsetM": (Sdf.ValueTypeNames.FloatArray, [0.0] * n),
        "omni:sensor:Core:emitterState:s001:vertOffsetM": (Sdf.ValueTypeNames.FloatArray, [0.0] * n),
        "omni:sensor:Core:numberOfChannels": (Sdf.ValueTypeNames.UInt, n),
        "omni:sensor:Core:numberOfEmitters": (Sdf.ValueTypeNames.UInt, n),
        # 1e9 / 55296 ns = 18084 firings/s; the RTX lidar needs an integer number of firings per revolution
        # (reportRate / scanRate), so use 18080 -> 1808 firings per revolution at 10 Hz (0.199 deg)
        "omni:sensor:Core:reportRateBaseHz": (Sdf.ValueTypeNames.UInt, 10 * round(1e9 / SEQUENCE_NS / 10)),
        "omni:sensor:Core:scanRateBaseHz": (Sdf.ValueTypeNames.UInt, 10),
        "omni:sensor:Core:scanType": (Sdf.ValueTypeNames.Token, "ROTARY"),
        "omni:sensor:Core:rotationDirection": (Sdf.ValueTypeNames.Token, "CW"),
        "omni:sensor:Core:startAzimuthOffsetDeg": (Sdf.ValueTypeNames.Float, 0.0),
        "omni:sensor:Core:nearRangeM": (Sdf.ValueTypeNames.Float, 0.9),
        "omni:sensor:Core:farRangeM": (Sdf.ValueTypeNames.Float, 100.0),
        "omni:sensor:Core:rangeAccuracyM": (Sdf.ValueTypeNames.Float, 0.03),
        "omni:sensor:Core:rangeResolutionM": (Sdf.ValueTypeNames.Float, 0.002),
        "omni:sensor:Core:azimuthErrorMean": (Sdf.ValueTypeNames.Float, 0.0),
        "omni:sensor:Core:azimuthErrorStd": (Sdf.ValueTypeNames.Float, 0.0),
        "omni:sensor:Core:elevationErrorMean": (Sdf.ValueTypeNames.Float, 0.0),
        "omni:sensor:Core:elevationErrorStd": (Sdf.ValueTypeNames.Float, 0.0),
        "omni:sensor:Core:maxReturns": (Sdf.ValueTypeNames.UInt, 1),
        "omni:sensor:Core:minReflectance": (Sdf.ValueTypeNames.Float, 0.1),
        "omni:sensor:Core:peakPowerW": (Sdf.ValueTypeNames.Float, 0.002),
        "omni:sensor:Core:pulseTimeNs": (Sdf.ValueTypeNames.UInt, 6),
        "omni:sensor:Core:intensityMappingType": (Sdf.ValueTypeNames.Token, "LINEAR"),
        "omni:sensor:Core:intensityProcessing": (Sdf.ValueTypeNames.Token, "NORMALIZATION"),
        "omni:sensor:Core:rayType": (Sdf.ValueTypeNames.Token, "IDEALIZED"),
        "omni:sensor:Core:accumulateOutputs": (Sdf.ValueTypeNames.Bool, True),
        "omni:sensor:Core:skipDroppingInvalidPoints": (Sdf.ValueTypeNames.Bool, False),
        "omni:sensor:modelName": (Sdf.ValueTypeNames.String, "Velodyne_VLP16"),
        "omni:sensor:tickRate": (Sdf.ValueTypeNames.Float, 10.0),
    }
    for name, (typ, val) in attrs.items():
        a = prim.CreateAttribute(name, typ)
        if isinstance(val, list):
            val = (Vt.FloatArray if typ == Sdf.ValueTypeNames.FloatArray else Vt.UIntArray)(val)
        a.Set(val)
    stage.GetRootLayer().Save()
    print(f"wrote {args.output}: {n} channels, {attrs['omni:sensor:Core:reportRateBaseHz'][1]} firings/s @ 10 Hz "
          f"-> {360 * 10 / attrs['omni:sensor:Core:reportRateBaseHz'][1]:.3f} deg azimuth step")


if __name__ == "__main__":
    main()
