"""Shared physical thresholds for the TD-MPC2 walker specifications."""

# Version 1 used 0.6 m, which is the point where dm_control's standing reward
# reaches its tolerance floor.  It means "standing badly", not "torso on the
# floor".  Version 2 measures the latter: nine passive-fall trials observed
# persistent floor--torso contact up to 0.2442132051 m.  The 0.27 m threshold
# adds a predeclared 0.02 m empirical margin and rounds upward to centimetres.
WALKER_GROUND_CONTACT_HEIGHT_MAX_OBSERVED_M = 0.2442132050751591
WALKER_FALL_HEIGHT_M = 0.27
WALKER_HEIGHT_SPEC_VERSION = "ground-contact-height-v2"

# A candidate posture core still needs to be recognisably upright.  This is a
# core-search condition and must not be confused with the fall predicate.
WALKER_UPRIGHT_CORE_HEIGHT_MIN_M = 1.0
