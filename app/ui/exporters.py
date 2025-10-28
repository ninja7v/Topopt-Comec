# app/analysis/exporters.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Export result to various file formats.

import mcubes
import numpy as np
import vtk
from stl import mesh
from vtk.util.numpy_support import get_vtk_array_type, numpy_to_vtk


# class Exporters:
def save_as_vti(xPhys: np.ndarray, nelxyz: list, filename: str):
    """Saves the density field as a .vti file for ParaView."""
    try:
        if len(nelxyz) == 3 and nelxyz[2] > 0:
            nx, ny, nz = nelxyz
            density_field = xPhys.reshape((nz, nx, ny))
        else:
            nx, ny = nelxyz[:2]
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
        return True, None  # Success, no error
    except Exception as e:
        return False, str(e)  # Failure, error message


def save_as_stl(xPhys: np.ndarray, nelxyz: list, filename: str):
    """Saves the result as a solid .stl file."""
    try:
        if len(nelxyz) == 3 and nelxyz[2] > 0:
            nx, ny, nz = nelxyz
            density_field = xPhys.reshape((nz, nx, ny))
        else:
            nx, ny = nelxyz[:2]
            nz = 1  # Extrude to a single layer
            # Reshape 2D data and add a new axis for the Z dimension
            density_field = xPhys.reshape((nx, ny)).T[np.newaxis, :, :]

        # Common Marching Cubes and STL Logic
        vertices, triangles = mcubes.marching_cubes(density_field, 0.5)
        stl_mesh = mesh.Mesh(np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype))
        stl_mesh.vectors = vertices[triangles]
        stl_mesh.save(filename)
        return True, None
    except Exception as e:
        return False, str(e)
