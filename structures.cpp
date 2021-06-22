#ifndef STRUCTURES
#define STRUCTURES

/* STRUCTURES FILE
 *
 * Vec, Ray, Sphere
 * 
 * Defines Material types
 * 
 */
#include <math.h>

// Vec Structure
struct Vec
{
    // Position Variables
    double x, y, z;
    __host__ __device__ Vec(){}
    __host__ __device__ ~Vec(){}

    // Constructor
    __host__ __device__ Vec(double x_, double y_, double z_)
        : x(x_), y(y_), z(z_) {}

    // Operator functions:
    __host__ __device__ Vec operator+(const Vec &other) const
    {
        return Vec(x + other.x, y + other.y, z + other.z);
    }
    __host__ __device__ Vec operator-(const Vec &other) const
    {
        return Vec(x - other.x, y - other.y, z - other.z);
    }
    __host__ __device__ Vec operator*(double factor) const
    {
        return Vec(x * factor, y * factor, z * factor);
    }
    __host__ __device__ Vec mult(const Vec &other) const
    {
        return Vec(x * other.x, y * other.y, z * other.z);
    }
    // Normalize:
    __host__ __device__ Vec &norm()
    {
        return *this = *this * (1 / sqrt(x * x + y * y + z * z));
    }
    // Dot Product:
    __host__ __device__ double dot(const Vec &other) const
    {
        return (x * other.x) + (y * other.y) + (z * other.z);
    }
    // Cross Product:
    __host__ __device__ Vec operator%(Vec &other)
    {
        double x_ = y * other.z - z * other.y;
        double y_ = z * other.x - x * other.z;
        double z_ = x * other.y - y * other.x;
        return Vec(x_, y_, z_);
    }
};

// Ray Structure
struct Ray
{
    // Origin and Direction Vectors
    Vec o, d;
    // Constructor
    __host__ __device__ Ray(Vec o_, Vec d_)
        : o(o_), d(d_) {}
    //Default Constructor
    __host__ __device__ Ray()
    {
        o = Vec();
        d = Vec();
    }
};

// Material Types Defined - (DIFFuse, SPECular, REFRactive)
enum Refl_t
{
    DIFF,
    SPEC,
    REFR
};

// Sphere Structure
struct Sphere
{
    // Radius
    double rad;
    // Position, Emission, & Color
    Vec pos, emission, color;
    // Reflection Type
    Refl_t refl;
    // Constructor
    __host__ __device__ Sphere(double rad_, Vec pos_, Vec emission_, Vec color_, Refl_t refl_)
        : rad(rad_), pos(pos_), emission(emission_), color(color_), refl(refl_) {}

    //Default constructor
    __host__ __device__ Sphere()
    {
        //constructor must be empty for constant memory initialization
        // rad = 0;
        // pos = Vec();
        // emission = Vec();
        // color = Vec();
        // refl = DIFF;
    }

    __host__ __device__ ~Sphere(){}

    __host__ __device__ Sphere & operator= (const Sphere& other){ 
        rad = other.rad;
        pos = other.pos;
        emission = other.emission;
        color = other.color;
        refl = other.refl;
        return *this;
    }

    // Intersect Function
    //      - Returns distance if intersect, 0 if not
    __host__ __device__ double intersect(const Ray &ray) const
    {
        /* Vector Sphere Equation: (P-C) . (P-C) - r^2 = 0
         * Ray Equation:                 P = O + tD
         *               (D.D)t^2 + 2D.(O-C)t + (O-C).(O-C) - r^2 = 0
         * Solve w/ Quadratic Formula:
         *                      t = (-b +- sqrt(b^2 - 4ac)) / 2a
         *              a = D.D     b = 2D.(O-C)    c = (O-C).(O-C) - r^2
         * 
         * (a = 1 because ray is normalized & the 2s and 4 will all cancel out)
         */
        Vec op = pos-ray.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        double t;
        double eps=1e-4;
        double b=op.dot(ray.d);
        double det=b*b-op.dot(op)+rad*rad;
        if (det<0) return 0; else det=sqrt(det);
        return (t=b-det)>eps ? t : ((t=b+det)>eps ? t : 0);
    }
};

typedef double my_type;

/*      Initial intersect code... just in case...
Vec op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
double t, eps = 1e-4, b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;
if (det < 0)
    return 0;
else
    det = sqrt(det);
return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
*/

#endif
