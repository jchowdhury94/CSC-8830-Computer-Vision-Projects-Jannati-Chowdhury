"""Stereo-assisted room (wall) measurement from two overlapping stereo video views -- Module 7 Part 3, Phase 1.

Method J: one shared stereo rig pose estimated jointly from all frame sets (robust epipolar + rigid four-view track
reprojection), then floor reconstruction, boundary-line corner inference, two-view registration and width/height.

Ground-truth dimensions live ONLY in ``room_measurement.validation`` and are never imported by the estimation code.
"""
__all__ = ["calibration", "geometry", "frames", "features", "pose", "lines", "measure", "uncertainty", "pipeline", "validation"]
