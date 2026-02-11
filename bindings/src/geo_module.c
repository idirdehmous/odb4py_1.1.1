#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <float.h>
#define PYPRINT(o)  PyObject_Print(o, stdout, 0); printf("\n");

#define POWDI(x,i) pow(x,i)

void sp_gcdist(double *lon1, double *lon2, double *lat1, double *lat2, double *dist)
{
    double F, G, L, sinG2, cosG2, sinF2, cosF2, sinL2, cosL2, S, C;
    double w, R, a, f, D, H1, H2;
    double lat1R, lat2R, lon1R, lon2R, DE2RA;

    DE2RA = M_PI / 180.0;
    a = 6378.137;              /* WGS-84 radius (km) */
    f = 1.0 / 298.257223563;   /* flattening */

    /* quick return if coordinates identical */
    if (fabs(lat1[0] - lat2[0]) < DBL_EPSILON &&
        fabs(fmod(lon1[0] - lon2[0], 360.0)) < DBL_EPSILON) {
        dist[0] = 0.0;
        return;
    }

    lat1R = lat1[0] * DE2RA;
    lat2R = lat2[0] * DE2RA;
    lon1R = lon1[0] * DE2RA;
    lon2R = lon2[0] * DE2RA;

    F = (lat1R + lat2R) / 2.0;
    G = (lat1R - lat2R) / 2.0;
    L = (lon1R - lon2R) / 2.0;

    sinG2 = sin(G) * sin(G);
    cosG2 = cos(G) * cos(G);
    sinF2 = sin(F) * sin(F);
    cosF2 = cos(F) * cos(F);
    sinL2 = sin(L) * sin(L);
    cosL2 = cos(L) * cos(L);

    S = sinG2 * cosL2 + cosF2 * sinL2;
    C = cosG2 * cosL2 + sinF2 * sinL2;

    /* safeguard */
    if (C <= 0.0 || S < 0.0) {
        dist[0] = 0.0;
        return;
    }

    w = atan(sqrt(S / C));
    if (w == 0.0) {
        dist[0] = 0.0;
        return;
    }

    R = sqrt(S * C) / w;
    D = 2.0 * w * a;
    H1 = (3.0 * R - 1.0) / (2.0 * C);
    H2 = (3.0 * R + 1.0) / (2.0 * S);

    dist[0] = D * (1.0 + f * (H1 * sinF2 * cosG2 - H2 * cosF2 * sinG2));
}




void sp_dists(double *u, double *v, double *uout, double *vout, int *n, double *dists, int *lonlat) {
        int N = *n, j;
        double gc[1];   

        if (lonlat[0] == 0) {
                for (j = 0; j < N; j++)
                        dists[j] = hypot(u[j] - uout[0], v[j] - vout[0]);
        } else {
                for (j = 0; j < N; j++) {
                        sp_gcdist(u+j, uout, v+j, vout, gc);
                        dists[j] = gc[0];
                }
        }
}



static PyObject* odbGcdistance_method( PyObject* Py_UNUSED(self) , PyObject* args)
{
    import_array()
    PyObject      *lon1_obj  , *lat1_obj , *lon2_obj, *lat2_obj;
    PyArrayObject *lon1      , *lat1     , *lon2    , *lat2    ;


    if (!PyArg_ParseTuple(args, "OOOO", &lon1_obj,
			                &lat1_obj,
					&lon2_obj,
					&lat2_obj)){ 
        
	PyErr_SetString(PyExc_RuntimeError, "--odb4py : Failed to parse odbGcdistance arguments ");
        return NULL;  
    }

   
   if ( !lon1_obj || !lat1_obj || !lon2_obj  || !lat2_obj  ) {
      PyErr_SetString(PyExc_RuntimeError, "--odb4py : 4 Arguments are required");
      return NULL;    
   }



    // Parse args and DECREF if it fails 
    lon1 = (PyArrayObject*)PyArray_FROM_OTF( lon1_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!lon1) {  
	PyErr_SetString(PyExc_RuntimeError, "--odb4py : Failed to parse lon1");    
	return NULL;
    }

    lat1 = (PyArrayObject*)PyArray_FROM_OTF(lat1_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!lat1) {
       Py_DECREF(lon1);
       PyErr_SetString(PyExc_RuntimeError, "--odb4py : Failed to parse lat1");
       return NULL;
       }

    lon2 = (PyArrayObject*)PyArray_FROM_OTF(lon2_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!lon2) {
       Py_DECREF(lon1);
       Py_DECREF(lat1);
       PyErr_SetString(PyExc_RuntimeError, "--odb4py : Failed to parse lon2");
       return NULL;
      }


    lat2 = (PyArrayObject*)PyArray_FROM_OTF( lat2_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!lat2) {
    Py_DECREF(lon1);
    Py_DECREF(lat1);
    Py_DECREF(lon2);
    PyErr_SetString(PyExc_RuntimeError, "--odb4py : Failed to parse lat2");
    return NULL;
    }


    // Set dimensions  [n x n ] matrix    
    int n1 = (int)PyArray_DIM(lat1, 0);
    int n2 = (int)PyArray_DIM(lat2, 0);
    
    npy_intp dims[2] = {n1, n2};
    PyArrayObject *distmat = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    if (!distmat) {
    Py_DECREF(lat1);
    Py_DECREF(lon1);
    Py_DECREF(lat2);
    Py_DECREF(lon2);
    PyErr_SetString(PyExc_RuntimeError, "--odb4py : Unable to allocate result for the contigous matrix.");
    return NULL;
    }
    
    double *latdata1 = (double*)PyArray_DATA(lat1);
    double *londata1 = (double*)PyArray_DATA(lon1);
    double *latdata2 = (double*)PyArray_DATA(lat2);
    double *londata2 = (double*)PyArray_DATA(lon2);
    double *distdata = (double*)PyArray_DATA(distmat);

    for (int i = 0; i < n1; ++i)
        for (int j = 0; j < n2; ++j) {
            double d;
            sp_gcdist(&londata1[i], &londata2[j], &latdata1[i], &latdata2[j], &d);
            distdata[i * n2 + j] = d;
        }
    
    Py_DECREF(lat1);
    Py_DECREF(lon1);
    Py_DECREF(lat2);
    Py_DECREF(lon2);
    return (PyObject*)distmat;
}
