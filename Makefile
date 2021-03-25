.PHONY: src/mkl.rs
src/mkl.rs:
	bindgen ---rust-target nightly --size_t-is-usize \
	--whitelist-function mkl_cblas_jit_create_sgemm \
	--whitelist-function mkl_jit_get_sgemm_ptr \
	--whitelist-function mkl_jit_destroy \
	--whitelist-function sgemm_direct \
	--whitelist-function MKL_Set_Threading_Layer \
	--whitelist-var MKL_THREADING_SEQUENTIAL \
	/opt/intel/compilers_and_libraries_2020.4.304/linux/mkl/include/mkl.h \
	-- -D MKL_ILP64 -D MKL_DIRECT_CALL_SEQ > src/mkl.rs
