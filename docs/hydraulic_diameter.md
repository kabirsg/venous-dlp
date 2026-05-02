The hydraulic diameter script was designed to calculate the hydraulic diameter, area, and perimeter of the vessel at each centerline point. 

This script requires that surface_prep and one of the centerline generation files in Geometry Tools to be run already.

The load_centerline function gets the relevant values from a centerline vtp file and returns it.

The cross_sections_metric function calculates the perimeter, area, and hydraulic diameter at each centerline point.
The methods used by this function are the following:
|Function| Return Type| Use |
|---|---|---|
|`trimesh.section(plane_origin=origin, plane_normal=normal)`| Path3D object [1] | Creates a plane using the given plane_origin point and plane_normal. Finds all points that intersects this plane and creates a line connecting each of these points. |

[1] Path3D is an object with two values: 
    vertices: A (n, 3) numpy array of points in global 3D space
    entities: A list of "Line" objects. Each line is a pair of indices referring back to the vertices (eg. Line(points=[0, 1]))