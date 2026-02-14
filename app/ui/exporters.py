# app/ui/exporters.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Export result to various file formats.

import mcubes
import matplotlib.pyplot as plt
import numpy as np
import vtk
import trimesh
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from stl import mesh
from vtk.util.numpy_support import get_vtk_array_type, numpy_to_vtk


def save_as_png(xPhys: np.ndarray, nelxyz: list, filename: str):
    """Saves the density field as a .png image."""
    try:
        nelx, nely, nelz = nelxyz
        is_3d = nelz > 0

        fig = plt.figure()
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor("white")
            # Avoids plotting fully transparent points.
            visible_elements_mask = xPhys > 0.01
            visible_indices = np.where(visible_elements_mask)[0]
            densities = xPhys[visible_indices]

            z = visible_indices // (nelx * nely)
            x = (visible_indices % (nelx * nely)) // nely
            y = visible_indices % nely

            colors = np.zeros((len(densities), 4))
            base_color_rgb = to_rgb("black")  # Default to black for CLI export
            colors[:, :3] = base_color_rgb
            colors[:, 3] = densities

            ax.scatter(
                x + 0.5,
                y + 0.5,
                z + 0.5,
                s=6000 / max(nelx, nely, nelz),
                marker="s",
                c=colors,
                alpha=None,
            )
            ax.set_box_aspect([nelx, nely, nelz])
            # Hide axes for cleaner output
            ax.set_axis_off()
        else:
            ax = fig.add_subplot(111)
            ax.set_facecolor("white")
            cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white", "black"])
            ax.imshow(
                xPhys.reshape((nelx, nely)).T,
                cmap=cmap,
                interpolation="nearest",
                origin="lower",
                norm=plt.Normalize(0, 1),
            )
            ax.set_aspect("equal", "box")
            ax.axis("off")

        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return True, None
    except Exception as e:
        return False, str(e)


def save_as_vti(xPhys: np.ndarray, nelxyz: list, filename: str):
    """Saves the density field as a .vti file for ParaView."""
    try:
        nx, ny, nz = nelxyz
        if nz > 0:
            density_field = xPhys.reshape((nz, nx, ny)).transpose(1, 2, 0)
        else:
            nz = 1  # Extrude to a single layer
            # Reshape 2D data and add a new axis for the Z dimension
            density_field = xPhys.reshape((nx, ny))[np.newaxis, :, :]

        # VTK requires data to be flattened in Fortran order ('F')
        vtk_array = numpy_to_vtk(
            num_array=density_field.flatten("F"),
            deep=True,
            array_type=get_vtk_array_type(density_field.dtype),
        )

        image_data = vtk.vtkImageData()
        image_data.SetOrigin([0, 0, 0])
        image_data.SetSpacing([1, 1, 1])
        image_data.SetDimensions([nx, ny, nz])
        image_data.GetPointData().SetScalars(vtk_array)
        image_data.GetPointData().GetScalars().SetName("Density")

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(image_data)
        writer.Write()
        return True, None
    except Exception as e:
        return False, str(e)


def save_as_stl(xPhys: np.ndarray, nelxyz: list, filename: str):
    """Saves the result as a .stl file."""
    try:
        nx, ny, nz = nelxyz
        if nz > 0:
            density_field = xPhys.reshape((nz, nx, ny)).transpose(1, 2, 0)
        else:
            nz = 1  # Extrude to a single layer
            # Reshape 2D data and add a new axis for the Z dimension
            density_field = xPhys.reshape((nx, ny)).T[np.newaxis, :, :]

        # Add 1-voxel padding to avoid border loss in marching cubes
        density_field = np.pad(
            density_field, pad_width=1, mode="constant", constant_values=0
        )
        # Run marching cubes
        vertices, triangles = mcubes.marching_cubes(density_field, 0.5)
        # Build STL mesh
        stl_mesh = mesh.Mesh(np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype))
        stl_mesh.vectors = vertices[triangles]
        stl_mesh.save(filename)
        return True, None
    except Exception as e:
        return False, str(e)


def save_as_3mf(xPhys: np.ndarray, nelxyz: list, filename: str):
    """Saves the result as a .3mf file using Trimesh."""
    try:
        nx, ny, nz = nelxyz
        if nz > 0:
            density_field = xPhys.reshape((nz, nx, ny)).transpose(1, 2, 0)
        else:
            nz = 1  # Extrude to a single layer
            # Reshape 2D data and add a new axis for the Z dimension
            density_field = xPhys.reshape((nx, ny)).T[np.newaxis, :, :]

        # Add 1-voxel padding to avoid border loss in marching cubes
        density_field = np.pad(
            density_field, pad_width=1, mode="constant", constant_values=0
        )
        # Run marching cubes
        vertices, triangles = mcubes.marching_cubes(density_field, 0.5)

        # Build 3MF mesh
        mesh_3mf = trimesh.Trimesh(vertices=vertices, faces=triangles)

        # Trimesh handles the complex 3MF XML/Zip structure automatically
        mesh_3mf.export(filename, file_type="3mf")

        return True, None
    except Exception as e:
        return False, str(e)
