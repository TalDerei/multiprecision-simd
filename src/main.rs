use multiprecision_simd::mont1;
// use multiprecision_simd::run;

fn main() {
    println!("Main function is running.");
    crate::mont1::benchmark_bm17_simd_mont_mul();
}
