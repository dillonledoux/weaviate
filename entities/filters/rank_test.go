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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateRank(t *testing.T) {
	tests := []struct {
		name    string
		rank    *Rank
		wantErr bool
		errMsg  string
	}{
		{
			name:    "nil rank returns no error",
			rank:    nil,
			wantErr: false,
		},
		{
			name:    "empty conditions returns error",
			rank:    &Rank{Conditions: []RankCondition{}},
			wantErr: true,
			errMsg:  "at least one condition is required",
		},
		{
			name: "negative weight returns error",
			rank: &Rank{
				Weight: -1.0,
				Conditions: []RankCondition{
					{Filter: &LocalFilter{}},
				},
			},
			wantErr: true,
			errMsg:  "weight must be between 0 and 1",
		},
		{
			name: "condition with none set returns error",
			rank: &Rank{
				Conditions: []RankCondition{
					{Weight: 1.0},
				},
			},
			wantErr: true,
			errMsg:  "exactly one of 'filter', 'decay', or 'property_value' must be set",
		},
		{
			name: "condition with both filter and decay returns error",
			rank: &Rank{
				Conditions: []RankCondition{
					{
						Filter: &LocalFilter{},
						Decay:  &Decay{Path: &Path{Property: "age"}, Origin: "30", Scale: "10"},
					},
				},
			},
			wantErr: true,
			errMsg:  "exactly one of 'filter', 'decay', or 'property_value' must be set",
		},
		{
			name: "condition with negative weight is valid",
			rank: &Rank{
				Conditions: []RankCondition{
					{Filter: &LocalFilter{}, Weight: -0.5},
				},
			},
			wantErr: false,
		},
		{
			name: "decay with missing path returns error",
			rank: &Rank{
				Conditions: []RankCondition{
					{Decay: &Decay{Scale: "10"}},
				},
			},
			wantErr: true,
			errMsg:  "path is required",
		},
		{
			name: "decay with missing origin is valid (defaults to now for dates)",
			rank: &Rank{
				Conditions: []RankCondition{
					{Decay: &Decay{Path: &Path{Property: "created_at"}, Scale: "7d"}},
				},
			},
			wantErr: false,
		},
		{
			name: "decay with missing scale returns error",
			rank: &Rank{
				Conditions: []RankCondition{
					{Decay: &Decay{Path: &Path{Property: "age"}, Origin: "30"}},
				},
			},
			wantErr: true,
			errMsg:  "scale is required",
		},
		{
			name: "decay with invalid curve returns error",
			rank: &Rank{
				Conditions: []RankCondition{
					{Decay: &Decay{
						Path:   &Path{Property: "age"},
						Origin: "30",
						Scale:  "10",
						Curve:  "invalid",
					}},
				},
			},
			wantErr: true,
			errMsg:  "curve must be one of",
		},
		{
			name: "negative depth returns error",
			rank: &Rank{
				Depth: -1,
				Conditions: []RankCondition{
					{Filter: &LocalFilter{}, Weight: 1.0},
				},
			},
			wantErr: true,
			errMsg:  "depth must be >= 0",
		},
		{
			name: "positive depth is valid",
			rank: &Rank{
				Depth: 500,
				Conditions: []RankCondition{
					{Filter: &LocalFilter{}, Weight: 1.0},
				},
			},
			wantErr: false,
		},
		{
			name: "property_value with missing path returns error",
			rank: &Rank{
				Conditions: []RankCondition{
					{PropertyValue: &PropertyValue{Modifier: "log1p"}},
				},
			},
			wantErr: true,
			errMsg:  "path is required",
		},
		{
			name: "property_value with invalid modifier returns error",
			rank: &Rank{
				Conditions: []RankCondition{
					{PropertyValue: &PropertyValue{
						Path:     &Path{Property: "likes"},
						Modifier: "invalid",
					}},
				},
			},
			wantErr: true,
			errMsg:  "modifier must be one of",
		},
		{
			name: "valid property_value condition",
			rank: &Rank{
				Conditions: []RankCondition{
					{PropertyValue: &PropertyValue{
						Path:     &Path{Property: "likes"},
						Modifier: "log1p",
					}},
				},
			},
			wantErr: false,
		},
		{
			name: "condition with filter and property_value returns error",
			rank: &Rank{
				Conditions: []RankCondition{
					{
						Filter:        &LocalFilter{},
						PropertyValue: &PropertyValue{Path: &Path{Property: "likes"}},
					},
				},
			},
			wantErr: true,
			errMsg:  "exactly one of",
		},
		{
			name: "valid filter condition returns no error",
			rank: &Rank{
				Weight: 0.5,
				Conditions: []RankCondition{
					{Filter: &LocalFilter{}, Weight: 1.0},
				},
			},
			wantErr: false,
		},
		{
			name: "valid decay condition returns no error",
			rank: &Rank{
				Weight: 0.5,
				Conditions: []RankCondition{
					{Decay: &Decay{
						Path:   &Path{Property: "age"},
						Origin: "30",
						Scale:  "10",
						Curve:  "exp",
					}, Weight: 1.0},
				},
			},
			wantErr: false,
		},
		{
			name: "multiple valid conditions returns no error",
			rank: &Rank{
				Weight: 0.8,
				Conditions: []RankCondition{
					{Filter: &LocalFilter{}, Weight: 1.0},
					{Decay: &Decay{
						Path:   &Path{Property: "timestamp"},
						Origin: "now",
						Scale:  "7d",
						Curve:  "gauss",
					}, Weight: 2.0},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateRank(tt.rank)
			if tt.wantErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.errMsg)
			} else {
				require.NoError(t, err)
			}
		})
	}
}
