"""Minimal SE(3) oracle helpers for external lab adapters.

Convention used by `abs_pose_error_6d`:

    T_error = inv(T_estimate) @ T_ground_truth

The returned vector is absolute [tx, ty, tz, rx, ry, rz], where translation is
in metres and the rotation vector is in radians. Laboratories may use another
predeclared convention, but must not mix conventions across splits.
"""
from __future__ import annotations
import math
import numpy as np

def rotation_vector(R):
    R=np.asarray(R,float)
    c=max(-1.0,min(1.0,(float(np.trace(R))-1.0)/2.0))
    theta=math.acos(c)
    if theta < 1e-10:
        return np.zeros(3)
    if abs(math.pi-theta) < 1e-7:
        A=(R+np.eye(3))/2.0
        axis=np.sqrt(np.maximum(np.diag(A),0.0))
        if R[2,1]-R[1,2] < 0: axis[0]*=-1
        if R[0,2]-R[2,0] < 0: axis[1]*=-1
        if R[1,0]-R[0,1] < 0: axis[2]*=-1
        n=np.linalg.norm(axis)
        if n < 1e-12: raise ValueError("rotation log is numerically undefined near pi")
        return axis/n*theta
    v=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/(2.0*math.sin(theta))
    return v*theta

def validate_transform(T):
    T=np.asarray(T,float)
    if T.shape != (4,4): raise ValueError("transform must be 4x4")
    if not np.all(np.isfinite(T)): raise ValueError("transform contains non-finite values")
    if not np.allclose(T[3],[0,0,0,1],atol=1e-8): raise ValueError("invalid homogeneous last row")
    R=T[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=1e-5) or np.linalg.det(R) < 0.999:
        raise ValueError("rotation block is not a proper rotation")
    return T

def invert(T):
    T=validate_transform(T); R=T[:3,:3]; t=T[:3,3]
    out=np.eye(4); out[:3,:3]=R.T; out[:3,3]=-R.T@t
    return out

def pose_error_6d(T_estimate,T_ground_truth):
    Te=invert(validate_transform(T_estimate)) @ validate_transform(T_ground_truth)
    return np.r_[Te[:3,3],rotation_vector(Te[:3,:3])]

def abs_pose_error_6d(T_estimate,T_ground_truth):
    return np.abs(pose_error_6d(T_estimate,T_ground_truth))
