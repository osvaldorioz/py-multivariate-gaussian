#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>

//c++ -O2 -mavx -shared -std=c++20 -fPIC $(python3.12 -m pybind11 --includes) multivariate_gaussian.cpp -o multivariate_gaussian$(python3.12-config --extension-suffix)

namespace py = pybind11;

class MultivariateGaussian {
public:
    MultivariateGaussian(const std::vector<double> &mean, const std::vector<std::vector<double>> &covariance)
        : mean_(mean), covariance_(covariance), dim_(mean.size()) {
        if (mean.size() != covariance.size() || covariance.size() != covariance[0].size()) {
            throw std::invalid_argument("Dimensiones incompatibles entre la media y la covarianza.");
        }
        compute_inverse_and_determinant();
    }

    // Evaluar la PDF en un punto dado
    double pdf(const std::vector<double> &x) const {
        if (x.size() != dim_) {
            throw std::invalid_argument("El tamaño de x debe coincidir con la dimensión de la distribución.");
        }
        std::vector<double> diff(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            diff[i] = x[i] - mean_[i];
        }

        double exponent = 0.0;
        for (size_t i = 0; i < dim_; ++i) {
            for (size_t j = 0; j < dim_; ++j) {
                exponent += diff[i] * inv_covariance_[i][j] * diff[j];
            }
        }
        exponent = -0.5 * exponent;

        double normalization = 1.0 / std::sqrt(std::pow(2 * M_PI, dim_) * det_covariance_);
        return normalization * std::exp(exponent);
    }

    // Generar muestras aleatorias
    py::array_t<double> sample(int n_samples) const {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, 1.0);

        // Crear una matriz para almacenar las muestras
        std::vector<std::vector<double>> samples(n_samples, std::vector<double>(dim_, 0.0));

        // Generar muestras estándar
        std::vector<double> Z(dim_);
        for (int s = 0; s < n_samples; ++s) {
            for (size_t i = 0; i < dim_; ++i) {
                Z[i] = dist(gen);
            }

            // Transformar las muestras estándar
            for (size_t i = 0; i < dim_; ++i) {
                for (size_t j = 0; j < dim_; ++j) {
                    samples[s][i] += covariance_[i][j] * Z[j];
                }
                samples[s][i] += mean_[i];
            }
        }

        // Convertir las muestras a py::array_t
        //py::array_t<double> result({n_samples, dim_});
        py::array_t<double> result(py::array::ShapeContainer({n_samples, dim_}));
        auto result_buffer = result.request();
        double *result_ptr = static_cast<double *>(result_buffer.ptr);

        for (int i = 0; i < n_samples; ++i) {
            for (size_t j = 0; j < dim_; ++j) {
                result_ptr[i * dim_ + j] = samples[i][j];
            }
        }

        return result;
    }

private:
    std::vector<double> mean_;
    std::vector<std::vector<double>> covariance_;
    std::vector<std::vector<double>> inv_covariance_;
    double det_covariance_;
    size_t dim_;

    // Calcular la inversa y el determinante de la matriz de covarianza
    void compute_inverse_and_determinant() {
        inv_covariance_ = covariance_;
        size_t n = dim_;
        det_covariance_ = 1.0;

        // Matriz identidad
        std::vector<std::vector<double>> identity(n, std::vector<double>(n, 0.0));
        for (size_t i = 0; i < n; ++i) {
            identity[i][i] = 1.0;
        }

        // Gauss-Jordan para invertir la matriz
        for (size_t i = 0; i < n; ++i) {
            double diag_element = inv_covariance_[i][i];
            if (diag_element == 0.0) {
                throw std::runtime_error("La matriz de covarianza no es invertible.");
            }
            det_covariance_ *= diag_element;

            for (size_t j = 0; j < n; ++j) {
                inv_covariance_[i][j] /= diag_element;
                identity[i][j] /= diag_element;
            }

            for (size_t k = 0; k < n; ++k) {
                if (k != i) {
                    double factor = inv_covariance_[k][i];
                    for (size_t j = 0; j < n; ++j) {
                        inv_covariance_[k][j] -= factor * inv_covariance_[i][j];
                        identity[k][j] -= factor * identity[i][j];
                    }
                }
            }
        }
        inv_covariance_ = identity;
    }
};

// Enlace Pybind11
PYBIND11_MODULE(multivariate_gaussian, m) {
    py::class_<MultivariateGaussian>(m, "MultivariateGaussian")
        .def(py::init<const std::vector<double> &, const std::vector<std::vector<double>> &>())
        .def("pdf", &MultivariateGaussian::pdf)
        .def("sample", &MultivariateGaussian::sample);
}
