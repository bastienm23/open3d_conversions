#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ros_numpy
import rospy
import numpy as np
import open3d
import copy
from sensor_msgs.msg import PointCloud2, PointField


def from_msg(ros_cloud):
    xyzrgb_array = ros_numpy.point_cloud2.pointcloud2_to_array(ros_cloud)

    mask = (
        np.isfinite(xyzrgb_array["x"])
        & np.isfinite(xyzrgb_array["y"])
        & np.isfinite(xyzrgb_array["z"])
    )
    cloud_array = xyzrgb_array[mask]

    open3d_cloud = open3d.geometry.PointCloud()

    points = np.zeros(cloud_array.shape + (3,), dtype=np.float)
    points[..., 0] = cloud_array["x"]
    points[..., 1] = cloud_array["y"]
    points[..., 2] = cloud_array["z"]
    open3d_cloud.points = open3d.utility.Vector3dVector(points)

    if "rgb" in xyzrgb_array.dtype.names:
        rgb_array = ros_numpy.point_cloud2.split_rgb_field(xyzrgb_array)
        cloud_array = rgb_array[mask]

        colors = np.zeros(cloud_array.shape + (3,), dtype=np.float)
        colors[..., 0] = cloud_array["r"]
        colors[..., 1] = cloud_array["g"]
        colors[..., 2] = cloud_array["b"]

        open3d_cloud.colors = open3d.utility.Vector3dVector(colors / 255.0)

    if "rgba" in xyzrgb_array.dtype.names:
        rgb_array = split_rgba_field(xyzrgb_array)
        cloud_array = rgb_array[mask]

        colors = np.zeros(cloud_array.shape + (3,), dtype=np.float)
        colors[..., 0] = cloud_array["r"]
        colors[..., 1] = cloud_array["g"]
        colors[..., 2] = cloud_array["b"]

        open3d_cloud.colors = open3d.utility.Vector3dVector(colors / 255.0)

    return open3d_cloud


BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8


def to_msg(open3d_cloud, frame_id=None, stamp=None):
    """convert open3d point cloud to ros point cloud
    Args:
        o3dpc (open3d.geometry.PointCloud): open3d point cloud
        frame_id (string): frame id of ros point cloud header
        stamp (rospy.Time): time stamp of ros point cloud header
    Returns:
        rospc (sensor.msg.PointCloud2): ros point cloud message
    """

    cloud_npy = np.asarray(copy.deepcopy(open3d_cloud.points))
    is_color = open3d_cloud.colors

    n_points = len(cloud_npy[:, 0])
    if is_color:
        data = np.zeros(
            n_points,
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32),
            ],
        )
    else:
        data = np.zeros(
            n_points,
            dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)],
        )
    data["x"] = cloud_npy[:, 0]
    data["y"] = cloud_npy[:, 1]
    data["z"] = cloud_npy[:, 2]

    if is_color:
        rgb_npy = np.asarray(copy.deepcopy(open3d_cloud.colors))
        rgb_npy = np.floor(rgb_npy * 255)  # nx3 matrix
        rgb_npy = (
            rgb_npy[:, 0] * BIT_MOVE_16
            + rgb_npy[:, 1] * BIT_MOVE_8
            + rgb_npy[:, 2]
        )
        rgb_npy = rgb_npy.astype(np.uint32)
        data["rgb"] = rgb_npy

    rospc = ros_numpy.msgify(PointCloud2, data)
    if frame_id is not None:
        rospc.header.frame_id = frame_id

    if stamp is None:
        rospc.header.stamp = rospy.Time.now()
    else:
        rospc.header.stamp = stamp
    rospc.height = 1
    rospc.width = n_points
    rospc.fields = []
    rospc.fields.append(
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1)
    )
    rospc.fields.append(
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1)
    )
    rospc.fields.append(
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
    )

    if is_color:
        rospc.fields.append(
            PointField(
                name="rgb", offset=12, datatype=PointField.UINT32, count=1
            )
        )
        rospc.point_step = 16
    else:
        rospc.point_step = 12

    rospc.is_bigendian = False
    rospc.row_step = rospc.point_step * n_points
    rospc.is_dense = True
    return rospc


def split_rgba_field(cloud_arr):
    """Takes an array with a named 'rgb' float32 field, and returns an array in which
    this has been split into 3 uint 8 fields: 'r', 'g', and 'b'.

    (pcl stores rgb in packed 32 bit floats)
    """
    rgb_arr = cloud_arr["rgba"].copy()
    rgb_arr.dtype = np.uint32
    r = np.asarray((rgb_arr >> 16) & 255, dtype=np.uint8)
    g = np.asarray((rgb_arr >> 8) & 255, dtype=np.uint8)
    b = np.asarray(rgb_arr & 255, dtype=np.uint8)

    # create a new array, without rgb, but with r, g, and b fields
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if not field_name == "rgb":
            new_dtype.append((field_name, field_type))
    new_dtype.append(("r", np.uint8))
    new_dtype.append(("g", np.uint8))
    new_dtype.append(("b", np.uint8))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:
        if field_name == "r":
            new_cloud_arr[field_name] = r
        elif field_name == "g":
            new_cloud_arr[field_name] = g
        elif field_name == "b":
            new_cloud_arr[field_name] = b
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]
    return new_cloud_arr
