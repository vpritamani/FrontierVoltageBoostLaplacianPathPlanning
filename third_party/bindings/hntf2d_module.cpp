// pybind11 binding over maxchaos/hntf2d (GPL-3.0) HarmonicMap2D.
//
// Upstream wraps the class with Cython through a customised eigency fork;
// pybind11's built-in Eigen support exposes the same C++ methods without that
// extra dependency. The C++ implementation (cpp/src/map.cpp) is compiled
// unmodified from the pinned upstream checkout.

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <hntf2d/map.h>

namespace py = pybind11;

PYBIND11_MODULE(external_hntf2d, m) {
  m.doc() = "maxchaos/hntf2d harmonic map from a multiply connected planar "
            "domain onto the punctured unit disk";

  py::class_<HarmonicMap2D>(m, "HarmonicMap2D")
      .def(py::init<>())
      .def("boundary_append", &HarmonicMap2D::boundary_append, py::arg("mtx"),
           "Append a closed boundary polyline (K x 2, first row == last row). "
           "The first boundary is the outer one (counter-clockwise); the rest "
           "are obstacles (clockwise).")
      .def("boundary_get", &HarmonicMap2D::boundary_get, py::arg("index"))
      .def("boundaries_amount", &HarmonicMap2D::boundaries_amount)
      .def("nodes_amount", &HarmonicMap2D::nodes_amount)
      .def("solve", &HarmonicMap2D::solve, py::arg("uo"), py::arg("vo"),
           py::call_guard<py::gil_scoped_release>(),
           "Solve the panel-method system given the images (uo, vo) of the "
           "outer boundary's segment centres on the unit circle.")
      .def("map", &HarmonicMap2D::map, py::arg("x"), py::arg("y"))
      .def("jacob", &HarmonicMap2D::jacob, py::arg("x"), py::arg("y"))
      .def("puncts", &HarmonicMap2D::puncts,
           "Images (u_i, v_i) of the inner boundaries (one row per obstacle).")
      // Panel-method system matrices (diagnostics).
      .def("mtx_geom", &HarmonicMap2D::mtx_geom)
      .def("mtx_univ", &HarmonicMap2D::mtx_univ)
      .def("mtx_aug", &HarmonicMap2D::mtx_aug);
}
