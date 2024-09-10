mod utils;

use crate::utils::gen_seeded_rng;
use num_bigint::{BigUint, RandomBits};
use num_traits::Pow;
use rand::Rng;
use std::ops::{Add, Mul};
use wasm_bindgen_test::*;
use web_sys::console;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

/// These test functions implement the algorithms describe in EZW18:
/// "Faster Modular Exponentiation Using Double Precision Floating Point Arithmetic on the GPU"
/// by Niall Emmart, Fangyu Zheng, and Charles Weems
/// https://ieeexplore.ieee.org/document/8464792/similar#similar

/////////////////////////////////////////////////////////////////// Utility Functions ///////////////////////////////////////////////////////////////////

/// Fused-multiply-add (FMA) is a floating-point multiply–add operation: (self * a) + b with a single rounding error:
/// https://doc.rust-lang.org/std/primitive.f64.html#method.mul_add
///
/// It computes the entire expression before rounding the result down to `N` signifigant bits.
/// Under the hood, it's implemented using LLVM intrinsics, specifically `llvm.fma.f64`
/// (https://github.com/rust-lang/rust/blob/38c943f7efc4ac76efc1611a702e903ff8ad67b2/src/librustc_trans/trans/intrinsic.rs#L62).
/// Performance is contigent on a CPU that supports the FMA instruction.
fn fma_64(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

fn fma_32(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

/// Algorithm 2.
///
/// Computes the high and low parts of the product of two floating-point values using FMA.
/// This is technique is used to minimize rounding errors. The specific algorithm is flawed
/// because the results suffer from alignment and round-off problems.
fn full_product_fp64(a: f64, b: f64) -> (f64, f64) {
    let p_hi = fma_64(a, b, 0.0); // p_hi = a * b + 0
    let p_lo = fma_64(a, b, -p_hi); // p_lo = a * b - p_hi
    (p_hi, p_lo)
}

fn full_product_fp32(a: f32, b: f32) -> (f32, f32) {
    let p_hi = fma_32(a, b, 0.0);
    let p_lo = fma_32(a, b, -p_hi);
    (p_hi, p_lo)
}

/// Algorithm 3.
///
/// Shows how adding 2^104 to the high product and subtracting 2^104 + 2^52 to the
/// low product "forces alignment of the decimal places". Note that this is algorithm is not
/// suitable for platforms that do not have the equivalent of the `__fma_rz()` CUDA function.
fn dpf_full_product(a: f64, b: f64) -> (f64, f64) {
    let c1: f64 = 2f64.powi(104i32);
    let c2: f64 = 2f64.powi(104i32).add(2f64.powi(52i32));
    let c3: f64 = 2f64.powi(52i32);

    let p_hi = fma_64(a, b, c1); // Computes the high part (a * b + 2^104)
    let sub = c2 - p_hi; // Aligns the low part with (2^52)
    let p_lo = fma_64(a, b, sub); // Computes the low part, accounting for precision loss

    (p_hi - c1, p_lo - c3) // Normalize and return the high and low parts
}

/// Algorithm 4.
///
/// It does not always work correctly on platforms which do not give the
/// developer control over the rounding mode.
fn int_full_product(a: f64, b: f64) -> (u64, u64) {
    let c1: f64 = 2f64.powi(104i32);
    let c2: f64 = 2f64.powi(104i32).add(2f64.powi(52i32));
    let mask = 2u64.pow(52) - 1; // 52-bit mask

    let p_hi = fma_64(a, b, c1);
    let sub = c2 - p_hi;
    let p_lo = fma_64(a, b, sub);

    (p_hi.to_bits() & mask, p_lo.to_bits() & mask)
}

fn print_f64(label: &str, a: f64) {
    let bits = a.to_bits();

    // Extract the sign bit (most significant bit)
    let sign = (bits >> 63) & 1;

    // Extract the exponent bits (next 11 bits)
    let exponent = ((bits >> 52) & 0x7FF) as i16;

    // Subtract the bias (1023) from the exponent
    let unbiased_exponent = exponent - 1023;

    // Extract the mantissa bits (last 52 bits)
    let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;

    // Print the components
    console::log_1(
        &format!(
            "{}: ({}, {}, 0x{:013X})",
            label, sign, unbiased_exponent, mantissa
        )
        .into(),
    );
}

/// Niall Emmart's Zprize 2023 implementation for performing multiprecision arithmetic with 51-bit limbs with SIMD FMA.
/// This function modifies `int_full_product()` in EZW18 with his technique.
///
/// Adapted from https://github.com/z-prize/2023-entries/blob/a568b9eb1e73615f2cee6d0c641bb6a7bc5a428f/
/// prize-2-msm-wasm/prize-2b-twisted-edwards/yrrid-snarkify/yrrid/SupportingFiles/FP51.java#L4.
fn niall_full_product(a: f64, b: f64) -> (u64, u64) {
    let c1: f64 = 2f64.powi(103i32); // c1 = 2^103
    let c2: f64 = c1.add(2f64.powi(51i32).mul(3f64)); // c2 = 2^103 + 3 * 2^51
    let tt: f64 = 2f64.powi(51i32).mul(3f64); // tt = 2^51 * 3
    let mask = 2u64.pow(51) - 1; // mask = 2^51 - 1

    let mut hi = fma_64(a, b, c1);
    let sub = c2 - hi;
    let mut lo = fma_64(a, b, sub);

    let mut hi = hi.to_bits() - c1.to_bits();
    let mut lo = lo.to_bits() - tt.to_bits();

    // If the lower word is negative, subtract 1 from the higher word
    if lo & 0x8000000000000000u64 > 0 {
        hi -= 1;
    } else {
    }

    lo = lo & mask;

    (hi, lo)
}

/////////////////////////////////////////////////////////////////// Unit Tests ///////////////////////////////////////////////////////////////////

/// FP32 (IEEE-754), the mantissa is 23-bits of precision, meaning it can accurately represent
/// a product up to 2^23 without any rounding or precision loss. In FP64, the mantissa is 52-explicit
/// bits.
#[test]
#[wasm_bindgen_test]
fn sample_fp() {
    let a: f32 = 1000.0;
    let b: f32 = 1000.0;
    let shift_val = (1 << 23) as f32;
    let res: f32 = a * b + shift_val;

    // hexidecimal representation: 4B0F4240
    // binary representation: 0 10010110 00011110100001001000000
    // sign = positve
    // exponent = 150 - 127 = 2 << 23
    // mantissa = 2^-4 + 2^-5 + 2^-6 + 2^-7 + 2^-9 + 2^-14 + 2^-17
    // formula: sign * (1 + mantissa) * 2 ^ exponent = 9388608.00377

    // least signifigant bits of the float represent the product being computed.
    // the mantissa (fractional portion) represent the actual product (a * b).
    console::log_1(&format!("{:?}", res.to_bits()).into());
    console::log_1(&format!("{:08X}", 1001 * 1001).into());
}

#[test]
#[wasm_bindgen_test]
fn sample_fma() {
    let m = 10.0_f64;
    let x = 4.0_f64;
    let b = 60.0_f64;

    assert_eq!(fma_64(m, x, b), 100.0);

    let one_plus_eps = 1.0_f64 + f64::EPSILON;
    let one_minus_eps = 1.0_f64 - f64::EPSILON;
    let minus_one = -1.0_f64;

    console::log_1(&format!("one_plus_eps: {:?}", one_plus_eps).into());
    console::log_1(&format!("one_minus_eps: {:?}", one_minus_eps).into());
    console::log_1(&format!("minus_one: {:?}", minus_one).into());

    let negative_eps_squared_with_fma = fma_64(one_plus_eps, one_minus_eps, minus_one);
    console::log_1(
        &format!(
            "negative_eps_squared_with_fma: {:?}",
            negative_eps_squared_with_fma
        )
        .into(),
    );

    // the exact result of (1 + eps) * (1 - eps) = 1 - eps * eps
    assert_eq!(negative_eps_squared_with_fma, -f64::EPSILON * f64::EPSILON);

    let negative_eps_squared_without_fma = one_plus_eps * one_minus_eps + minus_one;
    console::log_1(
        &format!(
            "negative_eps_squared_without_fma: {:?}",
            negative_eps_squared_without_fma
        )
        .into(),
    );

    // different rounding with non-fused multiply and add
    assert_eq!(negative_eps_squared_without_fma, 0.0);
}

#[test]
#[wasm_bindgen_test]
fn fma_full_product() {
    let c0 = 1.23456789012345f32;
    let d0 = 9.87654321098765f32;
    let (p_hi_prime, p_lo_prime) = full_product_fp32(c0, d0);
    console::log_1(&format!("p_hi_prime: {:?}", p_hi_prime).into());
    console::log_1(&format!("p_lo_prime: {:?}", p_lo_prime).into());
    console::log_1(&format!("full product: {:?}", c0 * d0).into());

    let a0 = 2f64.powi(50);
    let a1 = 1f64;
    let b0 = 2f64.powi(50);
    let b1 = 1f64;

    // Compute the full products (high and low) for the first pair of values
    let p0 = full_product_fp64(a0, b0); // (2^100, 0)
    let p1 = full_product_fp64(a1, b1); // (1, 0)

    // Sum the high and low parts of p0 and p1
    let sum_hi = p0.0 + p1.0;
    let sum_lo = p0.1 + p1.1;

    // Expected high and low sums
    let expected_hi = 2f64.powi(100);
    let expected_lo = 1f64;

    // adding hi products and low products of p0 and p1, yields (2^100)
    // rather than the desired sum (2^100 + 1), due to alignment
    // and round-off problems.
    assert_eq!(sum_hi, expected_hi);
    assert_ne!(sum_lo, expected_lo);
}

#[test]
#[wasm_bindgen_test]
fn fma_dpf_full_product() {
    let a0 = 2f64.powi(50);
    let a1 = 1f64;
    let b0 = 2f64.powi(50);
    let b1 = 1f64;

    // Compute the full products (high and low) for the first pair of values
    let p0 = dpf_full_product(a0, b0); // (2^100, 0)
    let p1 = dpf_full_product(a1, b1); // (1, 0)

    // Sum the high and low parts of p0 and p1
    let sum_hi = p0.0 + p1.0;
    let sum_lo = p0.1 + p1.1;

    // Expected high and low sums
    let expected_hi = 2f64.powi(100);
    let expected_lo = 1f64;

    // adding hi products and low products of p0 and p1, yields (2^100, 1)
    assert_eq!(sum_hi, expected_hi);
    assert_eq!(sum_lo, expected_lo);

    let a = 73061669485388f64;
    let b = 571904368095603f64;
    let product = a * b;
    print_f64("product: ", product);
    let c1: f64 = 2f64.powi(103i32);
    console::log_1(&format!("a * b: {:064b}", (fma_64(a, b, 0.0)).to_bits()).into());
    console::log_1(&format!("c1: {:064b}", c1.to_bits()).into());

    let high_term = product + c1;
    let high_term_fma = fma_64(a, b, c1);
    console::log_1(&format!("high_term: {:064b}", high_term.to_bits()).into());
    console::log_1(&format!("high_term_fma: {:064b}", high_term_fma.to_bits()).into());
}

#[test]
#[wasm_bindgen_test]
fn test_niall_zprize() {
    let a = 730616694853888f64;
    let b = 571904368095603f64;

    let (hi, lo) = niall_full_product(a, b);
    let a = num_bigint::BigUint::from(a as u64);
    let b = num_bigint::BigUint::from(b as u64);
    let expected = &a * &b;

    let x = num_bigint::BigUint::from(2u32).pow(51u32);
    let s = x * hi + lo;

    assert_eq!(s, expected)
}
