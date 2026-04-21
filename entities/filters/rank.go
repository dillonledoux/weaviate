//                           _       _
// __      _____  __ ___   ___  __ _| |_ ___
// \ \ /\ / / _ \/ _` \ \ / / |/ _` | __/ _ \
//  \ V  V /  __/ (_| |\ V /| | (_| | ||  __/
//   \_/\_/ \___|\__,_| \_/ |_|\__,_|\__\___|
//
//  Copyright © 2016 - 2026 Weaviate B.V. All rights reserved.
//
//  CONTACT: hello@weaviate.io
//

package filters

import (
	"fmt"
)

// Rank represents a set of soft ranking conditions that promote or demote
// matching documents without excluding non-matching ones. Rank rescores the
// primary search results using a weighted combination of primary and rank scores.
type Rank struct {
	Conditions []RankCondition
	Weight     float32 // blending weight [0,1]: final = (1-w)*primary + w*rank, default 0.5
}

// RankCondition represents a single ranking condition. Exactly one of
// Filter or Decay must be set.
type RankCondition struct {
	Filter *LocalFilter // binary: 1 if match, 0 if not
	Decay  *Decay       // continuous: distance-based [0,1]
	Weight float32      // per-condition weight, default 1.0; negative values demote
}

// Decay defines a distance-based scoring function that produces a continuous
// score in [0,1] based on how far a property value is from an origin point.
type Decay struct {
	Path       *Path
	Origin     string  // "now", ISO date, or numeric string
	Scale      string  // "7d", "20" — required
	Offset     string  // default "0"
	Curve      string  // "exp" (default), "gauss", "linear"
	DecayValue float32 // score at scale distance, default 0.5
}

// MaxRankConditions is the maximum number of conditions allowed in a single
// rank clause. Each condition triggers index queries or per-document scoring,
// so an unbounded count could cause resource exhaustion.
const MaxRankConditions = 20

// ValidateRank validates a Rank struct for correctness. Returns nil for
// nil input.
func ValidateRank(rank *Rank) error {
	if rank == nil {
		return nil
	}

	if len(rank.Conditions) == 0 {
		return fmt.Errorf("rank: at least one condition is required")
	}

	if len(rank.Conditions) > MaxRankConditions {
		return fmt.Errorf("rank: too many conditions (%d), maximum is %d",
			len(rank.Conditions), MaxRankConditions)
	}

	if rank.Weight < 0 || rank.Weight > 1 {
		return fmt.Errorf("rank: weight must be between 0 and 1, got %f", rank.Weight)
	}

	for i, cond := range rank.Conditions {
		if err := validateRankCondition(cond, i); err != nil {
			return err
		}
	}

	return nil
}

func validateRankCondition(cond RankCondition, idx int) error {
	hasFilter := cond.Filter != nil
	hasDecay := cond.Decay != nil

	if !hasFilter && !hasDecay {
		return fmt.Errorf("rank condition[%d]: exactly one of 'filter' or 'decay' must be set", idx)
	}
	if hasFilter && hasDecay {
		return fmt.Errorf("rank condition[%d]: exactly one of 'filter' or 'decay' must be set, both are set", idx)
	}

	if hasFilter {
		if err := validateRankFilterOps(cond.Filter.Root, idx); err != nil {
			return err
		}
	}

	if hasDecay {
		return validateDecay(cond.Decay, idx)
	}

	return nil
}

// validateRankFilterOps checks that the filter only uses operators supported
// by in-memory evaluation. Unsupported operators would silently score 0.
func validateRankFilterOps(clause *Clause, condIdx int) error {
	if clause == nil {
		return nil
	}
	switch clause.Operator {
	case OperatorEqual, OperatorNotEqual,
		OperatorGreaterThan, OperatorGreaterThanEqual,
		OperatorLessThan, OperatorLessThanEqual,
		OperatorAnd, OperatorOr, OperatorNot,
		OperatorLike, OperatorIsNull:
		// supported
	case OperatorWithinGeoRange:
		return fmt.Errorf("rank condition[%d] filter: operator WithinGeoRange is not supported in rank conditions", condIdx)
	case ContainsAny, ContainsAll, ContainsNone:
		return fmt.Errorf("rank condition[%d] filter: operator %s is not supported in rank conditions", condIdx, clause.Operator.Name())
	}
	for i := range clause.Operands {
		if err := validateRankFilterOps(&clause.Operands[i], condIdx); err != nil {
			return err
		}
	}
	return nil
}

func validateDecay(d *Decay, condIdx int) error {
	if d.Path == nil {
		return fmt.Errorf("rank condition[%d] decay: path is required", condIdx)
	}
	if d.Origin == "" {
		return fmt.Errorf("rank condition[%d] decay: origin is required", condIdx)
	}
	if d.Scale == "" {
		return fmt.Errorf("rank condition[%d] decay: scale is required", condIdx)
	}

	switch d.Curve {
	case "exp", "gauss", "linear", "":
		// valid
	default:
		return fmt.Errorf("rank condition[%d] decay: curve must be one of 'exp', 'gauss', 'linear', got %q", condIdx, d.Curve)
	}

	// DecayValue == 0 is treated as unset (defaults to 0.5 at scoring time).
	// Explicit values must be in (0, 1].
	if d.DecayValue != 0 && (d.DecayValue < 0 || d.DecayValue > 1) {
		return fmt.Errorf("rank condition[%d] decay: decay_value must be between 0 and 1, got %f", condIdx, d.DecayValue)
	}

	return nil
}
