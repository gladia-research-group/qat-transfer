import Mathlib

noncomputable section

namespace ProofWorkspace.Final

section Proposition1

def lambdaStar (innerDR donorNormSq : Real) : Real :=
  innerDR / donorNormSq

def donorQuadratic (donorNormSq innerDR receiverNormSq lam : Real) : Real :=
  donorNormSq * lam ^ 2 - 2 * innerDR * lam + receiverNormSq

lemma donorQuadratic_complete_square
    (donorNormSq innerDR receiverNormSq lam : Real)
    (hDonor : donorNormSq ≠ 0) :
    donorQuadratic donorNormSq innerDR receiverNormSq lam =
      donorNormSq * (lam - lambdaStar innerDR donorNormSq) ^ 2 +
        (receiverNormSq - innerDR ^ 2 / donorNormSq) := by
  unfold donorQuadratic lambdaStar
  field_simp [hDonor]
  ring

theorem donorQuadratic_minimized_at_lambdaStar
    (donorNormSq innerDR receiverNormSq : Real)
    (hDonorPos : 0 < donorNormSq) :
    donorQuadratic donorNormSq innerDR receiverNormSq (lambdaStar innerDR donorNormSq)
      = receiverNormSq - innerDR ^ 2 / donorNormSq /\
      (forall lam : Real,
        donorQuadratic donorNormSq innerDR receiverNormSq
            (lambdaStar innerDR donorNormSq)
          <= donorQuadratic donorNormSq innerDR receiverNormSq lam) := by
  have hDonor : donorNormSq ≠ 0 := ne_of_gt hDonorPos
  constructor
  · have hsq :=
      donorQuadratic_complete_square donorNormSq innerDR receiverNormSq
        (lambdaStar innerDR donorNormSq) hDonor
    calc
      donorQuadratic donorNormSq innerDR receiverNormSq (lambdaStar innerDR donorNormSq)
          = donorNormSq * ((lambdaStar innerDR donorNormSq) - (lambdaStar innerDR donorNormSq)) ^ 2 +
              (receiverNormSq - innerDR ^ 2 / donorNormSq) := by
                simpa [lambdaStar] using hsq
      _ = receiverNormSq - innerDR ^ 2 / donorNormSq := by ring
  · intro lam
    have hlam :=
      donorQuadratic_complete_square donorNormSq innerDR receiverNormSq lam hDonor
    have hmin :=
      donorQuadratic_complete_square donorNormSq innerDR receiverNormSq
        (lambdaStar innerDR donorNormSq) hDonor
    have hnonneg :
        0 <= donorNormSq * (lam - lambdaStar innerDR donorNormSq) ^ 2 := by
      exact mul_nonneg hDonorPos.le (pow_two_nonneg _)
    have hmin_eq :
        donorQuadratic donorNormSq innerDR receiverNormSq (lambdaStar innerDR donorNormSq)
          = receiverNormSq - innerDR ^ 2 / donorNormSq := by
      calc
        donorQuadratic donorNormSq innerDR receiverNormSq (lambdaStar innerDR donorNormSq)
            = donorNormSq * ((lambdaStar innerDR donorNormSq) - (lambdaStar innerDR donorNormSq)) ^ 2 +
                (receiverNormSq - innerDR ^ 2 / donorNormSq) := by
                  simpa [lambdaStar] using hmin
        _ = receiverNormSq - innerDR ^ 2 / donorNormSq := by ring
    linarith [hlam, hmin_eq, hnonneg]

theorem recovered_fraction_eq_cosine_sq
    (innerDR donorNormSq receiverNormSq : Real)
    (hDonorPos : 0 < donorNormSq)
    (hReceiverPos : 0 < receiverNormSq) :
    ((1 / 2 : Real) * (innerDR ^ 2 / donorNormSq)) / ((1 / 2 : Real) * receiverNormSq)
      = innerDR ^ 2 / (donorNormSq * receiverNormSq) := by
  field_simp [hDonorPos.ne', hReceiverPos.ne']

theorem donor_transfer_helps_iff
    (innerDR donorNormSq receiverNormSq : Real)
    (hDonorPos : 0 < donorNormSq)
    (hReceiverPos : 0 < receiverNormSq) :
    0 < innerDR ^ 2 / (donorNormSq * receiverNormSq) ↔ innerDR ≠ 0 := by
  have hDenPos : 0 < donorNormSq * receiverNormSq := mul_pos hDonorPos hReceiverPos
  constructor
  · intro h
    have hNumPos : 0 < innerDR ^ 2 := by
      rcases (div_pos_iff).1 h with hcase | hcase
      · exact hcase.1
      · exfalso
        exact (not_lt_of_ge hDenPos.le) hcase.2
    exact (sq_pos_iff).1 hNumPos
  · intro h
    have hNumPos : 0 < innerDR ^ 2 := (sq_pos_iff).2 h
    exact div_pos hNumPos hDenPos

theorem donor_transfer_matches_qat_iff_eq_case
    (innerDR donorNormSq receiverNormSq : Real)
    (hDonorPos : 0 < donorNormSq)
    (hReceiverPos : 0 < receiverNormSq) :
    innerDR ^ 2 / (donorNormSq * receiverNormSq) = 1
      ↔ innerDR ^ 2 = donorNormSq * receiverNormSq := by
  have hDenNe : donorNormSq * receiverNormSq ≠ 0 := by
    exact mul_ne_zero hDonorPos.ne' hReceiverPos.ne'
  constructor
  · intro h
    have hmul := congrArg (fun t : Real => t * (donorNormSq * receiverNormSq)) h
    field_simp [hDenNe] at hmul
    simpa [mul_assoc, mul_comm, mul_left_comm] using hmul
  · intro h
    exact (div_eq_iff hDenNe).2 (by simp [h])

theorem donor_transfer_never_exceeds_qat
    (innerDR donorNormSq receiverNormSq : Real)
    (hDonorPos : 0 < donorNormSq)
    (hReceiverPos : 0 < receiverNormSq)
    (hCauchy : innerDR ^ 2 <= donorNormSq * receiverNormSq) :
    innerDR ^ 2 / (donorNormSq * receiverNormSq) <= 1 := by
  have hDenPos : 0 < donorNormSq * receiverNormSq := mul_pos hDonorPos hReceiverPos
  field_simp [hDenPos.ne']
  exact hCauchy

theorem donor_transfer_matches_qat_iff_multiple
    (innerDR donorNormSq receiverNormSq : Real)
    (isMultiple : Prop)
    (hDonorPos : 0 < donorNormSq)
    (hReceiverPos : 0 < receiverNormSq)
    (hEqCase : innerDR ^ 2 = donorNormSq * receiverNormSq ↔ isMultiple) :
    innerDR ^ 2 / (donorNormSq * receiverNormSq) = 1 ↔ isMultiple := by
  exact
    (donor_transfer_matches_qat_iff_eq_case innerDR donorNormSq receiverNormSq
      hDonorPos hReceiverPos).trans hEqCase

end Proposition1

end ProofWorkspace.Final
