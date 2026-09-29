import Mathlib

namespace ProofWorkspace.Final

section Proposition2

theorem proposition2_second_order_bookkeeping
    {c g0gR g0gDelta qR qDelta r0 rDelta L nR nDelta : Real}
    (hc0 : 0 <= c) (hc1 : c <= 1)
    (hsplit : g0gDelta = qR - qDelta + (r0 - rDelta))
    (hquad : qR - qDelta = c * qR)
    (hg0gR : g0gR = qR + r0)
    (hr0 : |r0| <= (L / 6) * nR ^ 3)
    (hrDelta : |rDelta| <= (L / 6) * nDelta ^ 3) :
    Exists (fun epsilon : Real =>
      g0gDelta = c * g0gR + epsilon /\
      |epsilon| <= (L / 6) * (nR ^ 3 + nDelta ^ 3)) := by
  refine Exists.intro ((1 - c) * r0 - rDelta) ?_
  constructor
  · calc
      g0gDelta = c * qR + (r0 - rDelta) := by nlinarith [hsplit, hquad]
      _ = c * (g0gR - r0) + (r0 - rDelta) := by nlinarith [hg0gR]
      _ = c * g0gR + ((1 - c) * r0 - rDelta) := by ring
  · have htri : |(1 - c) * r0 - rDelta| <= |(1 - c) * r0| + |rDelta| := by
      have h := abs_sub_le ((1 - c) * r0) 0 rDelta
      simpa using h
    have honec_nonneg : 0 <= 1 - c := by linarith
    have honec_abs_le_one : |1 - c| <= 1 := by
      have honec_le_one : 1 - c <= 1 := by linarith
      simpa [abs_of_nonneg honec_nonneg] using honec_le_one
    have hscaled : |(1 - c) * r0| <= |r0| := by
      calc
        |(1 - c) * r0| = |1 - c| * |r0| := by rw [abs_mul]
        _ <= 1 * |r0| := mul_le_mul_of_nonneg_right honec_abs_le_one (abs_nonneg r0)
        _ = |r0| := by ring
    have hstep : |(1 - c) * r0 - rDelta| <= |r0| + |rDelta| := by
      linarith [htri, hscaled]
    have hsum_bound :
        |r0| + |rDelta| <= (L / 6) * nR ^ 3 + (L / 6) * nDelta ^ 3 := by
      linarith [hr0, hrDelta]
    calc
      |(1 - c) * r0 - rDelta| <= |r0| + |rDelta| := hstep
      _ <= (L / 6) * nR ^ 3 + (L / 6) * nDelta ^ 3 := hsum_bound
      _ = (L / 6) * (nR ^ 3 + nDelta ^ 3) := by ring

end Proposition2

end ProofWorkspace.Final
