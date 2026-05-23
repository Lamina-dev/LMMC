function(add_lmmc_test target_name source_file)
	add_executable(${target_name} ${source_file})
	target_link_libraries(${target_name} PRIVATE lmmc)
	lmmc_copy_lammp_runtime(${target_name})
	add_test(NAME ${target_name} COMMAND ${target_name})
endfunction()

add_lmmc_test(lmmc_test_dense test_dense.c)
add_lmmc_test(lmmc_test_bicgstab test_bicgstab.c)
add_lmmc_test(lmmc_test_bicgstab_autogen test_bicgstab_autogen.c)
add_lmmc_test(lmmc_test_gmres test_gmres.c)
add_lmmc_test(lmmc_test_itersolve test_itersolve.c)
add_lmmc_test(lmmc_test_lu test_lu.c)
add_lmmc_test(lmmc_test_cholesky test_cholesky.c)
add_lmmc_test(lmmc_test_numeric test_numeric.c)
add_lmmc_test(lmmc_test_sparse test_sparse.c)
add_lmmc_test(lmmc_test_csc test_csc.c)
add_lmmc_test(lmmc_test_spgemm test_spgemm.c)
add_lmmc_test(lmmc_test_stats test_stats.c)
add_lmmc_test(lmmc_test_tensor test_tensor.c)
add_lmmc_test(lmmc_test_lammp_ext test_lammp_ext.c)
add_lmmc_test(lmmc_test_nonlinear test_nonlinear.c)
add_lmmc_test(lmmc_test_ode test_ode.c)
add_lmmc_test(lmmc_test_qr test_qr.c)
add_lmmc_test(lmmc_test_math_utils test_math_utils.c)
add_lmmc_test(lmmc_test_logging_ext test_logging_ext.c)
add_lmmc_test(lmmc_test_sparse_builder test_sparse_builder.c)
add_lmmc_test(lmmc_test_lammp_stats test_lammp_stats.c)
add_lmmc_test(lmmc_test_internal test_internal.c)
target_include_directories(lmmc_test_internal PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_lmmc_test(lmmc_test_dense_vec test_dense_vec.c)

add_lmmc_test(lmmc_test_dense_mat test_dense_mat.c)

add_lmmc_test(lmmc_test_dense_unit test_dense_unit.c)

add_lmmc_test(lmmc_test_sparse_coo test_sparse_coo.c)

add_lmmc_test(lmmc_test_interp_cspline test_interp_cspline.c)

add_lmmc_test(lmmc_test_sparse_utils test_sparse_utils.c)

add_lmmc_test(lmmc_test_interp_lagrange test_interp_lagrange.c)

add_lmmc_test(lmmc_test_eigen_sym test_eigen_sym.c)

add_lmmc_test(lmmc_test_eigen_general test_eigen_general.c)

add_lmmc_test(lmmc_test_eigen_sym_property test_eigen_sym_property.c)

add_lmmc_test(lmmc_test_eigen_gen_prop test_eigen_gen_prop.c)

add_lmmc_test(lmmc_test_eigen_sym_prop test_eigen_sym_prop.c)

add_lmmc_test(lmmc_test_svd test_svd.c)

add_lmmc_test(lmmc_test_random_dist test_random_dist.c)

add_lmmc_test(lmmc_test_sparse_cholesky test_sparse_cholesky.c)

add_lmmc_test(lmmc_test_random_rng test_random_rng.c)

add_lmmc_test(lmmc_test_quadrature test_quadrature.c)


add_lmmc_test(lmmc_test_interp_linear test_interp_linear.c)

add_lmmc_test(lmmc_test_precond test_precond.c)

add_lmmc_test(lmmc_test_init test_init.c)

add_lmmc_test(lmmc_test_linalg_extended test_linalg_extended.c)

add_lmmc_test(lmmc_test_eigen_extended test_eigen_extended.c)

add_lmmc_test(lmmc_test_itersolve_extended test_itersolve_extended.c)

add_lmmc_test(lmmc_test_dense_extended test_dense_extended.c)

add_lmmc_test(lmmc_test_interp_extended test_interp_extended.c)

add_lmmc_test(lmmc_test_nonlinear_extended test_nonlinear_extended.c)

add_lmmc_test(lmmc_test_ode_extended test_ode_extended.c)

add_lmmc_test(lmmc_test_convergence test_convergence.c)

add_lmmc_test(lmmc_test_memory_safety test_memory_safety.c)

add_lmmc_test(lmmc_test_numeric_extended test_numeric_extended.c)

add_lmmc_test(lmmc_test_stats_extended test_stats_extended.c)

add_lmmc_test(lmmc_test_random_extended test_random_extended.c)

add_lmmc_test(lmmc_test_sparse_extended test_sparse_extended.c)

add_lmmc_test(lmmc_test_tensor_extended test_tensor_extended.c)
