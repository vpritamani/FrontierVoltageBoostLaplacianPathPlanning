// pybind11 binding over thinks/fast-marching-method (MIT, header-only).
//
// The upstream repository ships its own binding (bindings/python), but it
// only instantiates 2D/3D and takes the grid as separate size/spacing
// arguments. This binding exposes the same solvers — SignedArrivalTime with
// the (HighAccuracy)VaryingSpeedEikonalSolver — for N = 2..6 so the planner
// wrapper can run on every map dimensionality the benchmark app generates.
//
// Index convention: the upstream Grid stores index[0] as the fastest-varying
// axis. A C-ordered numpy array of shape (s0, ..., s_{N-1}) therefore maps to
// an upstream grid of size (s_{N-1}, ..., s0), and an upstream cell index is
// the *reversed* numpy index — i.e. exactly the user (x, y, ...) convention
// used throughout this repo. Boundary indices are passed in that order.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "thinks/fast_marching_method/fast_marching_method.hpp"

namespace py = pybind11;
namespace fmm = thinks::fast_marching_method;

namespace {

using SpeedArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IndexArray = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using TimeArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

template <std::size_t N>
py::array_t<double> ArrivalTimeN(SpeedArray speed, IndexArray boundary_indices,
                                 TimeArray boundary_times, bool high_accuracy) {
  std::array<std::size_t, N> grid_size;
  for (std::size_t i = 0; i < N; ++i) {
    grid_size[i] = static_cast<std::size_t>(speed.shape(N - 1 - i));
  }
  auto const grid_spacing = fmm::detail::FilledArray<double, N>(1.0);

  auto const n_boundary = static_cast<std::size_t>(boundary_indices.shape(0));
  if (boundary_indices.ndim() != 2 ||
      static_cast<std::size_t>(boundary_indices.shape(1)) != N) {
    throw std::invalid_argument("boundary_indices must have shape (M, ndim)");
  }
  if (static_cast<std::size_t>(boundary_times.size()) != n_boundary) {
    throw std::invalid_argument("boundary_times must have length M");
  }

  auto idx = boundary_indices.unchecked<2>();
  std::vector<std::array<std::int32_t, N>> indices(n_boundary);
  for (std::size_t m = 0; m < n_boundary; ++m) {
    for (std::size_t d = 0; d < N; ++d) {
      indices[m][d] = idx(m, d);
    }
  }
  std::vector<double> times(boundary_times.data(),
                            boundary_times.data() + n_boundary);
  std::vector<double> speed_buffer(speed.data(), speed.data() + speed.size());

  std::vector<double> arrival;
  {
    py::gil_scoped_release release;
    if (high_accuracy) {
      auto const solver = fmm::HighAccuracyVaryingSpeedEikonalSolver<double, N>(
          grid_spacing, grid_size, speed_buffer);
      arrival = fmm::SignedArrivalTime(grid_size, indices, times, solver);
    } else {
      auto const solver = fmm::VaryingSpeedEikonalSolver<double, N>(
          grid_spacing, grid_size, speed_buffer);
      arrival = fmm::SignedArrivalTime(grid_size, indices, times, solver);
    }
  }

  std::vector<py::ssize_t> shape(speed.shape(), speed.shape() + N);
  py::array_t<double> out(shape);
  std::copy(arrival.begin(), arrival.end(), out.mutable_data());
  return out;
}

py::array_t<double> ArrivalTime(SpeedArray speed, IndexArray boundary_indices,
                                TimeArray boundary_times, bool high_accuracy) {
  switch (speed.ndim()) {
    case 2: return ArrivalTimeN<2>(speed, boundary_indices, boundary_times, high_accuracy);
    case 3: return ArrivalTimeN<3>(speed, boundary_indices, boundary_times, high_accuracy);
    case 4: return ArrivalTimeN<4>(speed, boundary_indices, boundary_times, high_accuracy);
    case 5: return ArrivalTimeN<5>(speed, boundary_indices, boundary_times, high_accuracy);
    case 6: return ArrivalTimeN<6>(speed, boundary_indices, boundary_times, high_accuracy);
    default:
      throw std::invalid_argument("fast marching binding supports 2-6 dimensions");
  }
}

}  // namespace

PYBIND11_MODULE(external_fmm, m) {
  m.doc() = "thinks/fast-marching-method varying-speed arrival times (N = 2..6)";
  m.def("arrival_time", &ArrivalTime, py::arg("speed"),
        py::arg("boundary_indices"), py::arg("boundary_times"),
        py::arg("high_accuracy") = true,
        R"pbdoc(
Signed arrival times on a unit-spaced grid.

speed            : ndarray, shape = grid shape, all values > 0
boundary_indices : int32 array (M, ndim), cell indices in (x, y, ...) order
boundary_times   : float array (M,), known arrival times at those cells
high_accuracy    : second-order (HighAccuracy*) solver when True
)pbdoc");
  m.attr("MAX_DIMS") = 6;
}
