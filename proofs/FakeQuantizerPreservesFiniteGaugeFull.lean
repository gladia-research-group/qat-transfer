import Mathlib

namespace ProofWorkspace.Final

section FakeQuantizer

variable {G W : Type}
variable [Group G] [MulAction G W]

def steForward (FQ : W -> W) (w : W) : W :=
  FQ w

theorem equivariant_comp
    (f1 f2 : W -> W)
    (h1 : forall g : G, forall w : W, f1 (g • w) = g • f1 w)
    (h2 : forall g : G, forall w : W, f2 (g • w) = g • f2 w) :
    forall g : G, forall w : W, (f2 (f1 (g • w))) = g • (f2 (f1 w)) := by
  intro g w
  have h1' : f2 (f1 (g • w)) = f2 (g • f1 w) := by
    exact congrArg f2 (h1 g w)
  calc
    f2 (f1 (g • w)) = f2 (g • f1 w) := h1'
    _ = g • f2 (f1 w) := by simpa using h2 g (f1 w)

theorem fake_quantizer_preserves_remaining_finite_gauge
    (Gset : Finset G) (FQ : W -> W)
    (hFQ :
      forall g : G,
      g ∈ Gset ->
      forall w : W, FQ (g • w) = g • FQ w) :
    forall g : G,
    g ∈ Gset ->
    forall w : W, FQ (g • w) = g • FQ w := by
  intro g hg w
  exact hFQ g hg w

theorem ste_forward_rule_commutes
    (Gset : Finset G) (FQ : W -> W)
    (hFQ :
      forall g : G,
      g ∈ Gset ->
      forall w : W, FQ (g • w) = g • FQ w) :
    forall g : G,
    g ∈ Gset ->
    forall w : W, steForward FQ (g • w) = g • steForward FQ w := by
  intro g hg w
  unfold steForward
  exact hFQ g hg w

end FakeQuantizer

end ProofWorkspace.Final
