import Mathlib

namespace ProofWorkspace.Final

section AdamW

theorem square_with_two_nonzero_entries_not_affine_in_tsq
    (a b alpha beta : Real) (ha : Ne a 0) (hb : Ne b 0) :
    Not (forall t : Real, (a * t + b) ^ 2 = alpha * t ^ 2 + beta) := by
  intro h
  have h1 := h 1
  have hm1 := h (-1)
  have hdiff : (a + b) ^ 2 = (-a + b) ^ 2 := by
    nlinarith [h1, hm1]
  have hab0 : a * b = 0 := by
    nlinarith [hdiff]
  have hz : Or (a = 0) (b = 0) := by
    exact mul_eq_zero.mp hab0
  cases hz with
  | inl hza => exact ha hza
  | inr hzb => exact hb hzb

theorem ratio_identity_forces_abs_eq_one
    (d eps : Real) (hd : Ne d 0) (heps : 0 < eps)
    (h :
      forall t : Real,
        0 <= t ->
          (1 / d) / (Real.sqrt t + eps) =
            d / (abs d * Real.sqrt t + eps)) :
    abs d = 1 := by
  have h0 := h 0 (by positivity)
  have h0' : (1 / d) / eps = d / eps := by
    simpa using h0
  have h1 := h0'
  field_simp [heps.ne', hd] at h1
  have hd2 : d ^ 2 = 1 := by
    nlinarith [h1]
  have habs_sq : (abs d) ^ 2 = 1 := by
    calc
      (abs d) ^ 2 = d ^ 2 := by simp
      _ = 1 := hd2
  nlinarith [habs_sq, abs_nonneg d]

theorem adamw_kills_continuous_linear_reparameterizations
    {idx : Type} (d : idx -> Real) (eps : Real)
    (hd : forall i, Ne (d i) 0) (heps : 0 < eps)
    (h_ratio :
      forall i : idx,
      forall t : Real,
        0 <= t ->
          (1 / d i) / (Real.sqrt t + eps) =
            d i / (abs (d i) * Real.sqrt t + eps)) :
    forall i, abs (d i) = 1 := by
  intro i
  exact ratio_identity_forces_abs_eq_one
    (d := d i)
    (eps := eps)
    (hd := hd i)
    (heps := heps)
    (h := by
      intro t ht
      simpa using h_ratio i t ht)

end AdamW

end ProofWorkspace.Final
