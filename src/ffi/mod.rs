
#[cfg(feature="mkl")]
#[allow(dead_code, non_camel_case_types, non_upper_case_globals, improper_ctypes, non_snake_case)]
pub mod mkl;

#[cfg(feature="blis")]
pub mod blis;
