// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <functional>
#include <tuple>

#include <Eigen/QR>

#include "frc/EigenCore.h"

namespace frc {

/**
 * Computes unscented transform of a set of sigma points and weights. CovDim
 * returns the mean and square-root covariance of the sigma points in a tuple.
 *
 * This works in conjunction with the UnscentedKalmanFilter class. For use with
 * square-root form UKFs.
 *
 * @tparam CovDim      Dimension of covariance of sigma points after passing
 *                     through the transform.
 * @tparam States      Number of states.
 * @param sigmas       List of sigma points.
 * @param Wm           Weights for the mean.
 * @param Wc           Weights for the covariance.
 * @param meanFunc     A function that computes the mean of 2 * States + 1 state
 *                     vectors using a given set of weights.
 * @param residualFunc A function that computes the residual of two state
 *                     vectors (i.e. it subtracts them.)
 * @param squareRootR  Square-root of the noise covariance of the sigma points.
 *
 * @return Tuple of x, mean of sigma points; S, square-root covariance of
 * sigmas.
 */
template <int CovDim, int States>
std::tuple<Vectord<CovDim>, Matrixd<CovDim, CovDim>>
SquareRootUnscentedTransform(
    const Matrixd<CovDim, 2 * States + 1>& sigmas,
    const Vectord<2 * States + 1>& Wm, const Vectord<2 * States + 1>& Wc,
    std::function<Vectord<CovDim>(const Matrixd<CovDim, 2 * States + 1>&,
                                  const Vectord<2 * States + 1>&)>
        meanFunc,
    std::function<Vectord<CovDim>(const Vectord<CovDim>&,
                                  const Vectord<CovDim>&)>
        residualFunc,
    const Matrixd<CovDim, CovDim>& squareRootR) {
  if constexpr (CovDim != Eigen::Dynamic && States != Eigen::Dynamic) {
    // New mean is usually just the sum of the sigmas * weights:
    //
    //      2n
    //   x̂ = Σ Wᵢ⁽ᵐ⁾𝒳ᵢ
    //      i=0
    //
    // equations (19) and (23) in the paper show this,
    // but we allow a custom function, usually for angle wrapping
    Vectord<CovDim> x = meanFunc(sigmas, Wm);

    // Form an intermediate matrix S⁻ as:
    //
    //   [√{W₁⁽ᶜ⁾}(𝒳_{1:2L} - x̂) √{Rᵛ}]
    //
    // the part of equations (20) and (24) within the "qr{}"
    Matrixd<CovDim, States * 2 + CovDim> Sbar;
    for (int i = 0; i < States * 2; i++) {
      Sbar.template block<CovDim, 1>(0, i) =
          std::sqrt(Wc[1]) *
          residualFunc(sigmas.template block<CovDim, 1>(0, 1 + i), x);
    }
    Sbar.template block<CovDim, CovDim>(0, States * 2) = squareRootR;

    // Compute the square-root covariance of the sigma points.
    //
    // We transpose S⁻ first because we formed it by horizontally
    // concatenating each part; it should be vertical so we can take
    // the QR decomposition as defined in the "QR Decomposition" passage
    // of section 3. "EFFICIENT SQUARE-ROOT IMPLEMENTATION"
    //
    // The resulting matrix R is the square-root covariance S, but it
    // is upper triangular, so we need to transpose it.
    //
    // equations (20) and (24)
    Matrixd<CovDim, CovDim> S = Sbar.transpose()
                                    .householderQr()
                                    .matrixQR()
                                    .template block<CovDim, CovDim>(0, 0)
                                    .template triangularView<Eigen::Upper>()
                                    .transpose();

    // Update or downdate the square-root covariance with (𝒳₀-x̂)
    // depending on whether its weight (W₀⁽ᶜ⁾) is positive or negative.
    //
    // equations (21) and (25)
    Eigen::internal::llt_inplace<double, Eigen::Lower>::rankUpdate(
        S, residualFunc(sigmas.template block<CovDim, 1>(0, 0), x), Wc[0]);

    return std::make_tuple(x, S);
  } else {
    const int covDim = sigmas.rows();
    const int states = (sigmas.cols() - 1) / 2;

    // New mean is usually just the sum of the sigmas * weights:
    //
    //      2n
    //   x̂ = Σ Wᵢ⁽ᵐ⁾𝒳ᵢ
    //      i=0
    //
    // equations (19) and (23) in the paper show this,
    // but we allow a custom function, usually for angle wrapping
    Eigen::VectorXd x = meanFunc(sigmas, Wm);

    // Form an intermediate matrix S⁻ as:
    //
    //   [√{W₁⁽ᶜ⁾}(𝒳_{1:2L} - x̂) √{Rᵛ}]
    //
    // the part of equations (20) and (24) within the "qr{}"
    Eigen::MatrixXd Sbar{covDim, 2 * states + covDim};
    for (int i = 0; i < States * 2; i++) {
      Sbar.col(i) = std::sqrt(Wc[1]) * residualFunc(sigmas.col(i + 1), x);
    }
    Sbar.block(0, states * 2, covDim, covDim) = squareRootR;

    // Compute the square-root covariance of the sigma points.
    //
    // We transpose S⁻ first because we formed it by horizontally
    // concatenating each part; it should be vertical so we can take
    // the QR decomposition as defined in the "QR Decomposition" passage
    // of section 3. "EFFICIENT SQUARE-ROOT IMPLEMENTATION"
    //
    // The resulting matrix R is the square-root covariance S, but it
    // is upper triangular, so we need to transpose it.
    //
    // equations (20) and (24)
    Eigen::MatrixXd S = Sbar.transpose()
                            .householderQr()
                            .matrixQR()
                            .block(0, 0, covDim, covDim)
                            .triangularView<Eigen::Upper>()
                            .transpose();

    // Update or downdate the square-root covariance with (𝒳₀-x̂)
    // depending on whether its weight (W₀⁽ᶜ⁾) is positive or negative.
    //
    // equations (21) and (25)
    Eigen::internal::llt_inplace<double, Eigen::Lower>::rankUpdate(
        S, residualFunc(sigmas.col(0), x), Wc[0]);

    return std::make_tuple(x, S);
  }
}

}  // namespace frc
