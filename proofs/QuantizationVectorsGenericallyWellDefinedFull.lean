import Mathlib

open MeasureTheory

namespace ProofWorkspace.Final

section WellDefined

variable {Theta Gauge Cell : Type}
variable [MeasurableSpace Theta]

def badGaugeSet (G : Finset Gauge) (badEq : Gauge -> Set Theta) : Set Theta :=
  Set.iUnion (fun i : {g // g ∈ G} => badEq i.1)

def boundarySet (C : Finset Cell) (bdry : Cell -> Set Theta) : Set Theta :=
  Set.iUnion (fun j : {c // c ∈ C} => bdry j.1)

def totalBadSet
    (G : Finset Gauge)
    (C : Finset Cell)
    (badEq : Gauge -> Set Theta)
    (bdry : Cell -> Set Theta) : Set Theta :=
  badGaugeSet G badEq ∪ boundarySet C bdry

theorem badGaugeSet_measure_zero
    (mu : Measure Theta)
    (G : Finset Gauge)
    (badEq : Gauge -> Set Theta)
    (hbad : forall g, g ∈ G -> mu (badEq g) = 0) :
    mu (badGaugeSet G badEq) = 0 := by
  unfold badGaugeSet
  refine MeasureTheory.measure_iUnion_null ?_
  intro i
  exact hbad i.1 i.2

theorem boundarySet_measure_zero
    (mu : Measure Theta)
    (C : Finset Cell)
    (bdry : Cell -> Set Theta)
    (hbdry : forall c, c ∈ C -> mu (bdry c) = 0) :
    mu (boundarySet C bdry) = 0 := by
  unfold boundarySet
  refine MeasureTheory.measure_iUnion_null ?_
  intro j
  exact hbdry j.1 j.2

theorem totalBadSet_measure_zero
    (mu : Measure Theta)
    (G : Finset Gauge)
    (C : Finset Cell)
    (badEq : Gauge -> Set Theta)
    (bdry : Cell -> Set Theta)
    (hbad : forall g, g ∈ G -> mu (badEq g) = 0)
    (hbdry : forall c, c ∈ C -> mu (bdry c) = 0) :
    mu (totalBadSet G C badEq bdry) = 0 := by
  have hGaugeZero : mu (badGaugeSet G badEq) = 0 := by
    exact badGaugeSet_measure_zero mu G badEq hbad
  have hBdryZero : mu (boundarySet C bdry) = 0 := by
    exact boundarySet_measure_zero mu C bdry hbdry
  unfold totalBadSet
  exact le_antisymm
    (le_trans (measure_union_le _ _) (by simp [hGaugeZero, hBdryZero]))
    (by positivity)

theorem quantization_vectors_generically_well_defined
    (mu : Measure Theta)
    (G : Finset Gauge)
    (C : Finset Cell)
    (badEq : Gauge -> Set Theta)
    (bdry : Cell -> Set Theta)
    (wellDefined : Theta -> Prop)
    (hbad : forall g, g ∈ G -> mu (badEq g) = 0)
    (hbdry : forall c, c ∈ C -> mu (bdry c) = 0)
    (hOutside :
      forall theta,
        theta ∉ totalBadSet G C badEq bdry ->
          wellDefined theta) :
    Filter.Eventually (fun theta => wellDefined theta) (MeasureTheory.ae mu) := by
  have hBadZero : mu (totalBadSet G C badEq bdry) = 0 := by
    exact totalBadSet_measure_zero mu G C badEq bdry hbad hbdry
  have hAlmostOutside :
      Filter.Eventually (fun theta => theta ∉ totalBadSet G C badEq bdry) (MeasureTheory.ae mu) := by
    exact (MeasureTheory.ae_iff).2 (by simpa using hBadZero)
  exact hAlmostOutside.mono (fun theta htheta => hOutside theta htheta)

end WellDefined

end ProofWorkspace.Final
