pub mod bigint;
pub mod mont;
pub mod mont1;
pub mod run;
pub mod utils;

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub fn main() {
    crate::run::do_run();
}
