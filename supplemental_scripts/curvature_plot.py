import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt


def rolling_average(curvature_array, window=7):
    '''
    Calculates the rolling average

    Return:
        - pd.core.series.Series of the Rolling average for every point
    '''
    #Work on a copy
    result = curvature_array.astype(float).copy()

    #1. Compute rolling average for valid positions
    weights = np.ones(window) / window
    valid_avg = np.convolve(curvature_array, weights, mode="valid")

    #2. Determine index offsets for left and right edges
    left_offset = window // 2
    right_offset = left_offset + len(valid_avg)

    #3. Replace only the interior values
    result[left_offset:right_offset] = valid_avg

    return result


def main():
    VTP = ""
    CASE_NAME = ""

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(VTP)
    reader.Update()
    polydata = reader.GetOutput()
    curvature_array_np = vtk_to_numpy(polydata.GetPointData().GetArray("Curvature"))
    curvature_array_np *= 10 #Adjusting for units: 1/mm -> 1/cm
    # curvature_pd = pd.Series(curvature_array_np)

    roll_1 = rolling_average(curvature_array_np, window=3) #1 point before and after
    roll_2 = rolling_average(curvature_array_np, window=5) #2 points before and after
    roll_3 = rolling_average(curvature_array_np, window=7) #3 points before and after
    roll_4 = rolling_average(curvature_array_np, window=9) #4 points before and after
    roll_5 = rolling_average(curvature_array_np, window=11) #5 points before and after

    #Plotting
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(curvature_array_np, color="black", label="Instantaneous")
    ax.plot(roll_1, color="#0000FF", label="1 Point Before & After")
    ax.plot(roll_2, color="#0F52BA", label="2 Points Before & After")
    ax.plot(roll_3, color="#0096FF", label="3 Points Before & After")
    ax.plot(roll_4, color="#87CEEB", label="4 Points Before & After")
    ax.plot(roll_5, color="#A7C7E7", label="5 Points Before & After")

    ax.set_xlabel("Centerline Point Index")
    ax.set_ylabel("Kappa Curvature value [1/cm]")
    ax.set_title("Rolling Average Comparison for Curvature")
    ax.legend(fontsize=10)

    plt.tight_layout()

    save_path = f"curvature_comp/{CASE_NAME}.png"
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()