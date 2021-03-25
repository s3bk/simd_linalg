fn main() {
    if std::env::var("CARGO_FEATURE_MKL").is_ok() {
        println!("cargo:rustc-link-search=native=/opt/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64");
        println!("cargo:rustc-link-lib=mkl_rt");
    }
}

