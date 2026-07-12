#include "cuda_lsmr/lsmr.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>

namespace py = pybind11;

namespace {

/// Owns one CUDA context, dual-format sparse matrix, and reusable LSMR workspace.
class SparseSession {
 public:
  /// Uploads a host CSC/CSR matrix and prepares both device storage formats.
  SparseSession(const std::string& format, std::int64_t rows, std::int64_t cols,
                py::array_t<std::int32_t, py::array::c_style> offsets,
                py::array_t<std::int32_t, py::array::c_style> indices,
                py::array_t<float, py::array::c_style> values)
      : context_(), solver_(context_) {
    if (offsets.ndim() != 1 || indices.ndim() != 1 || values.ndim() != 1 ||
        indices.size() != values.size()) {
      throw cuda_lsmr::Error("compressed sparse arrays must be flat and aligned");
    }
    const std::int64_t nnz = values.size();
    if (format == "csc") {
      if (offsets.size() != cols + 1) {
        throw cuda_lsmr::Error("CSC offsets length must equal cols + 1");
      }
      matrix_ = cuda_lsmr::SparseMatrix::FromCsc(
          context_, {rows, cols, nnz, offsets.data(), indices.data(), values.data(),
                     cuda_lsmr::MemoryLocation::kHost});
    } else if (format == "csr") {
      if (offsets.size() != rows + 1) {
        throw cuda_lsmr::Error("CSR offsets length must equal rows + 1");
      }
      matrix_ = cuda_lsmr::SparseMatrix::FromCsr(
          context_, {rows, cols, nnz, offsets.data(), indices.data(), values.data(),
                     cuda_lsmr::MemoryLocation::kHost});
    } else {
      throw cuda_lsmr::Error("format must be 'csc' or 'csr'");
    }
    context_.synchronize();
  }

  /// Solves from CuPy-compatible device pointers and synchronizes the result.
  cuda_lsmr::SolverResult SolveDevice(
      std::uintptr_t rhs_pointer, std::uintptr_t solution_pointer, float damp,
      float atol, float btol, float conlim, std::int32_t max_iterations,
      std::uintptr_t initial_guess_pointer) {
    cuda_lsmr::SolverOptions options{damp, atol, btol, conlim, max_iterations};
    cuda_lsmr::DenseVectorView initial_guess{};
    const cuda_lsmr::DenseVectorView* initial_guess_view = nullptr;
    if (initial_guess_pointer != 0) {
      initial_guess = {matrix_->cols(),
                       reinterpret_cast<const float*>(initial_guess_pointer),
                       cuda_lsmr::MemoryLocation::kDevice};
      initial_guess_view = &initial_guess;
    }
    auto result = solver_.solve(
        *matrix_,
        {matrix_->rows(), reinterpret_cast<const float*>(rhs_pointer),
         cuda_lsmr::MemoryLocation::kDevice},
        {matrix_->cols(), reinterpret_cast<float*>(solution_pointer)}, options,
        initial_guess_view);
    context_.synchronize();
    return result;
  }

  /// Returns matrix rows.
  [[nodiscard]] std::int64_t rows() const noexcept { return matrix_->rows(); }

  /// Returns matrix columns.
  [[nodiscard]] std::int64_t cols() const noexcept { return matrix_->cols(); }

  /// Returns matrix nonzeros.
  [[nodiscard]] std::int64_t nnz() const noexcept { return matrix_->nnz(); }

  /// Returns bytes reserved by the context CUDA memory pool.
  [[nodiscard]] std::size_t pool_reserved_bytes() const {
    return context_.pool_reserved_bytes();
  }

 private:
  cuda_lsmr::Context context_;
  cuda_lsmr::Solver solver_;
  std::unique_ptr<cuda_lsmr::SparseMatrix> matrix_;
};

}  // namespace

PYBIND11_MODULE(_cuda_lsmr, module) {
  module.doc() = "Zero-copy CuPy benchmark bindings for cuda-lsmr";
  module.attr("__version__") = cuda_lsmr::version();
  module.attr("minimum_cuda_version") = cuda_lsmr::minimum_cuda_version();

  py::class_<cuda_lsmr::SolverResult>(module, "SolverResult")
      .def_readonly("stop_code", &cuda_lsmr::SolverResult::stop_code)
      .def_readonly("iterations", &cuda_lsmr::SolverResult::iterations)
      .def_readonly("norm_r", &cuda_lsmr::SolverResult::norm_r)
      .def_readonly("norm_ar", &cuda_lsmr::SolverResult::norm_ar)
      .def_readonly("norm_a", &cuda_lsmr::SolverResult::norm_a)
      .def_readonly("cond_a", &cuda_lsmr::SolverResult::cond_a)
      .def_readonly("norm_x", &cuda_lsmr::SolverResult::norm_x);

  py::class_<SparseSession>(module, "SparseSession")
      .def(py::init<const std::string&, std::int64_t, std::int64_t,
                    py::array_t<std::int32_t, py::array::c_style>,
                    py::array_t<std::int32_t, py::array::c_style>,
                    py::array_t<float, py::array::c_style>>(),
           py::arg("format"), py::arg("rows"), py::arg("cols"),
           py::arg("offsets"), py::arg("indices"), py::arg("values"),
           "Upload CSC/CSR data and retain CSR(A) plus CSC(A) on the GPU.")
      .def("solve_device", &SparseSession::SolveDevice, py::arg("rhs_pointer"),
           py::arg("solution_pointer"), py::arg("damp") = 0.0F,
           py::arg("atol") = 1.0e-6F, py::arg("btol") = 1.0e-6F,
           py::arg("conlim") = 1.0e8F, py::arg("max_iterations") = 0,
           py::arg("initial_guess_pointer") = 0,
           "Solve from FP32 CUDA pointers; output storage belongs to the caller.")
      .def_property_readonly("rows", &SparseSession::rows)
      .def_property_readonly("cols", &SparseSession::cols)
      .def_property_readonly("nnz", &SparseSession::nnz)
      .def_property_readonly("pool_reserved_bytes",
                             &SparseSession::pool_reserved_bytes);
}
