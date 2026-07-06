#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/zero:0.6.1": num
#import "template.typ": academic-paper
// #show: ieee.with(
//   title: [Fast Iterative Solvers: Project 1],

//   authors: (
//     (
//       name: "Paul Budden",
//       department: [Matrikelnummer: 484284],
//     ),
//   ),
// )

#show: academic-paper.with(
  title: "Fast Iterative Solvers: Project 1",
  //abstract: [  ],
  authors: (
    (
      name: "Paul Budden",
      affiliation: "Matrikelnummer: 484284",
    ),
    (
      name: "Paula Winter",
      affiliation: "Matrikelnummer: 409827",
    ),
  ),
)
//#show image: none

= Introduction

The Task for this Project was to implement the General Minimum Residual (GMRES) Method for iteratively solving linear systems of equations, with

- Restarting after m iterations
- Jacobi preconditioning
- Gauss-Seidel preconditioning
- ILU(0) preconditioning

As well as the Conjugate Gradient Method without preconditioning.

= Theory
#lorem(120)

= Implementation
Both Methods were implemented in Python utilizing the sparse linear algebra functions of _scipy.sparse.linalg_.
The algorithms used in class were mostly followed, taking care to convert them to a zero indexed language.
Additionally statistics are being recorded for plotting, decreasing the performance of the algorithms.
A _pandas.Dataframe_ with time, residual, relative residual, error, iteration number and orthogonality in case of GMRES is returned by both functions.

Checks for the input matrices were implemented: positive-definiteness for GMRES and symmetric positive-definiteness for CG.


== GMRES


Restarted GMRES was implemented by repeatedly calling a inner _\_gmres_ function, completing m iterations of the algorithm.
Passing `m=-1` results in no restarting and consecutive building of the internal arrays for $H$ and $V$.
Otherwise arrays of length of the maximum iterations of the inner loop are created.

The left preconditioning $M^(-1) tilde(x)$ for GMRES was implemented as a function $f(tilde(x))$ on the RHS to allow generic preconditioning and the use of more efficient algorithms for GS and ILU(0) preconditioning using _scipy.sparse.linalg.spsolve_triangular_.

The ILU(0) factorization was implemented, allowing arbitrary patterns, but defaulting to a more efficient ILU(0).
The implementation creates dense matrices for in place modification of $A$ and lookup of the pattern.

The orthogonality check for GMRES computes $v_1 dot v_(m+1)$ instead of $v_1 dot v_m$, as the first iteration would otherwise compute $v_1 dot v_1 approx 1$, making the scaling in the plots difficult.

== CG

The CG method was implemented exactly like the pseudocode in class, taking care to only compute the matrix-vector-product $A p_m$ once per iteration.
Also no unnecessary loop variables are stored.

= GMRES Results


== Full GMRES Preconditioning

#figure(
  image("../out/GMRES/full_gmres_rel_res.png"),
  caption: [Relative Residual for full GMRES and the relative tolerance of $10^(-8)$.],
) <full_gmres_res>

Comparing the different preconditioners in plot @full_gmres_res, one can clearly see the benefits of preconditioning.
While the full GMRES method without preconditioning took 480 iterations to converge, using the preconditioners significantly decreased the amount of iterations needed.
The Jacobi and Gauss-Seidel preconditioned method produced similar results, Jacobi only needing 221 iterations and the Gauss-Seidel preconditioned  needing 162 iterations.
The ILU preconditioner performs exceptionally well, converging on the second iteration, even far surpassing the set tolerance.


== Full GMRES Orthogonaly
#figure(
  image("../out/GMRES/full_gmres_no_prec_orth.png"),
  caption: [Orthogonaly $v_1 dot v_(m+1)$ of the Krilov Vectors for full GMRES.],
) <full_gmres_orth>

For the full GMRES case, the orthogonality $v_1 dot v_(m+1)$ of the new Krilov Vector should remain $approx 0$.
At iteration `i=1` it does start off at machine precision and continuously gets worse (see @full_gmres_orth).
The high amount of floating point operations when orthogonalizing the new vector results in floating point errors, resulting in non orthogonal vectors.

== Influence of Restarting
#figure(
  image("../out/GMRES/gmres_t_m.png"),
  caption: [Runtime for different restart parameters m. `m=-1` corresponds to no restarting (`m=inf`).],
) <m_v_t>

#figure(
  image("../out/GMRES/gmres_krylov_m.png"),
  caption: [Amount of restarts, equalling to the amount of krylov spaces build for different restart parameters m. `m=-1` corresponds to no restarting (`m=inf`).],
) <m_v_krylov>

As expected the amount of krylov spaces grows with decreasing m (see @m_v_krylov).
An `m=1` means restarting every iteration and `m=-1=inf` means never restarting.

The analysis of the runtime in @m_v_t is not as clear as for the amount of iterations.
As the runtime is measured in an unoptimized implementation of the algorithm with logging, differences in runtime can not be clearly explained by only a difference in the parameter m.
Interestingly, dynamically increasing the array size for $V$ and $H$ (`m=-1`) is slightly faster than preallocating a bigger array once (`m=555`).

Keeping in mind the overhead of logging, one can say that the runtime does not significantly change for `m>200`.
There seems to be a minimum for `m=50` for this specific matrix for the not preconditioned case and Jacobi preconditioning.

#figure(
  image("../out/GMRES/gmres_matrix.png"),
  caption: [The values of the matrix stored in `gmres_matrix_msr.txt`.],
) <gmrs_matrix>

Comparing the runtimes of the different preconditioners, the ILU preconditioner is significantly faster than the others.
As the used matrix is diagonally dominant, the Gauss-Seidel preconditioner does not approximate the Matrix well by a triangular one (see @gmrs_matrix).
The Jacobi preconditioner therefore approximates the matrix very well, resulting in a great performance increase with little effort.

As previously discussed, the ILU preconditioner performs very well, converging on the second iteration.
The runtime needed to precompute the ILU is not included in these figures, so it could still very well be slower than the other methods in total.



== Preconditioning at `m=200`
#figure(
  image("../out/GMRES/gmres_200_rel_res.png"),
  caption: [Relative Residual for restarted GMRES (`m=200`) using different preconditioners and the relative tolerance of $10^(-8)$.],
) <gmres_200_res>

#figure(
  image("../out/GMRES/gmres_200_time.png"),
  caption: [Runtime for restarted GMRES (`m=200`) using different preconditioners.],
) <gmres_200_time>

The relative residual per iteration for `m=200` follows the same trend as for full GMRES.
Unconditioned GMRES takes 778 iterations while Jacobi takes 247 and Gauss-Seidel takes 162.
Again ILU preconditioning converges on the second iteration.


Looking at the runtime, the restarts every 200 iterations can be seen as kinks in the runtime plot.
As the time for a single iteration grows with the number of Krylov-Vectors, restarting lowers the time back down to the time for 1 Krylov-Vector.
The time for each iteration of unconditioned GMRES and Jacobi preconditioned GMRES seem to be equal, as the matrix vector multiplication of a diagonal matrix is quite cheap.

Gauss-Seidel Preconditioning needs more time per iteration as solving a triangular system is required every iteration.
The ILU preconditioning needs even more time per step, as two triangular systems must be solved, but that is offset by the small amount of iterations needed for convergence.

All Preconditioners result in a lower total runtime needed to converge compared to the unconditioned case.
As discussed before (also see @m_v_t), full GMRES takes the longest time, followed by Gauss-Seidel preconditioned GMRES, Jacobi preconditioning and ILU preconditioning.

The early restarts do not do i general not improve the runtime of the algorithm, even worsening the runtime at times for this specific matrix.
The reason the algorithm is still useful, is the memory footprint of the algorithm.
As the iterations increase, a very large matrix for the Krilov-Vectors and the H matrix might not fit into memory anymore, leading to either significant performance penalties swapping ram contents to disk, or not being able to run the algorithm at all.
Restarting lowers the memory footprint and a such prevents this.



// START TABLE

#let data = csv("../out/GMRES/gmres_stats.csv")
#let cols-to-keep = (2, 4, 10, 11, 9)
#let col_disp_names = ("2": "", "4": "m", "10": "t", "11": "i", "9": "r")

#let format-cell(row-idx, col-idx, value) = {
  if row-idx != 0 and (col-idx == 10 or col-idx == 9) {
    num(value, exponent: "eng", round: (mode: "figures", precision: 2), omit-unity-mantissa: true)
  } else if row-idx == 0 and col_disp_names.keys().contains(str(col-idx)) {
    col_disp_names.at(str(col-idx))
  } else {
    value
  }
}
#figure(
  table(
    columns: cols-to-keep.len(),
    // Add a light gray background to the header row
    //fill: (x, y) => if y == 0 { luma(230) } else { none },
    align: center,
    ..data
      .enumerate()
      .map(((row-idx, row)) => {
        cols-to-keep.map(col-idx => format-cell(row-idx, col-idx, row.at(col-idx)))
      })
      .flatten()
  ),
  caption: [GMRES Runs Metadata],
)
// END TABLE


= CG Results

Comparing the error and residuals between both given matrices, one can see that convergence for matrix 1 with 12022 iterations is way slower than for matrix 2 that converged after only 248 iterations.
The reason for that can be seen in the visualizations of the matrices  in @cg_matrix_1 and @cg_matrix_2.
While matrix 1 is diagonally dominant with very few off diagonal entries, matrix one also includes many off diagonal entries with a high magnitude.
This results in wider spread eigenvalues of the matrix an thus a slower convergence of the Conjugate Gradient Method.

Interestingly the relative and absolute residual norm for matrix 1 seem to oscillate after a high number of iterations.
The error norm does not fit that pattern.


#figure(
  image("../out/CG/cg_M1_combined.png"),
  caption: [Relative residual, absolute residual, error and the relative tolerance of $10^(-8)$ for matrix 1.],
) <cg_comb_1>

#figure(
  image("../out/CG/cg_matrix_M1.png"),
  caption: [The values of the matrix stored in `cg_matrix_msr_1.txt`.],
) <cg_matrix_1>

#figure(
  image("../out/CG/cg_M2_combined.png"),
  caption: [Relative residual, absolute residual, error and the relative tolerance of $10^(-8)$ for matrix 2.],
) <cg_comb_1>

#figure(
  image("../out/CG/cg_matrix_M2.png"),
  caption: [The values of the matrix stored in `cg_matrix_msr_2.txt`.],
) <cg_matrix_2>




// START TABLE

#let data = csv("../out/CG/CG_stats.csv")
#let cols-to-keep = (10, 7, 8, 9)
#let col_disp_names = ("10": "Dataset", "7": "r", "8": "t", "9": "i")

#let format-cell(row-idx, col-idx, value) = {
  if row-idx != 0 and (col-idx == 7 or col-idx == 8) {
    num(value, exponent: "eng", round: (mode: "figures", precision: 2), omit-unity-mantissa: true)
  } else if row-idx == 0 and col_disp_names.keys().contains(str(col-idx)) {
    col_disp_names.at(str(col-idx))
  } else {
    value
  }
}
#figure(
  table(
    columns: cols-to-keep.len(),
    // Add a light gray background to the header row
    //fill: (x, y) => if y == 0 { luma(230) } else { none },
    align: center,
    ..data
      .enumerate()
      .map(((row-idx, row)) => {
        cols-to-keep.map(col-idx => format-cell(row-idx, col-idx, row.at(col-idx)))
      })
      .flatten()
  ),
  caption: [CG Runs Metadata],
)

// END TABLE

= Conclusion
