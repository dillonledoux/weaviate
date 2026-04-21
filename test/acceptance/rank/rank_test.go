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

// Package rank_test contains acceptance tests for the "rank" soft-ranking
// feature. Rank rescores primary search results using ranking conditions,
// promoting or demoting matching documents without excluding non-matching ones.
//
// Running:
//
//	go test -count 1 -race -timeout 15m ./test/acceptance/rank/...
package rank_test

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/go-openapi/strfmt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/weaviate/weaviate/entities/models"
	"github.com/weaviate/weaviate/test/docker"
	"github.com/weaviate/weaviate/test/helper"
	graphqlhelper "github.com/weaviate/weaviate/test/helper/graphql"
)

const className = "Product"

var (
	id1 = strfmt.UUID("11111111-1111-1111-1111-111111111111")
	id2 = strfmt.UUID("22222222-2222-2222-2222-222222222222")
	id3 = strfmt.UUID("33333333-3333-3333-3333-333333333333")
	id4 = strfmt.UUID("44444444-4444-4444-4444-444444444444")
	id5 = strfmt.UUID("55555555-5555-5555-5555-555555555555")
)

func TestRank(t *testing.T) {
	ctx := context.Background()

	compose, err := docker.New().
		WithWeaviate().
		Start(ctx)
	require.NoError(t, err)
	defer func() {
		require.NoError(t, compose.Terminate(ctx))
	}()

	helper.SetupClient(compose.GetWeaviate().URI())
	defer helper.ResetClient()

	setupSchema(t)
	defer helper.DeleteClass(t, className)
	insertTestData(t)
	waitForIndexing(t)

	t.Run("filter-only rank", testFilterOnlyRank)
	t.Run("decay on numeric property", testDecayNumeric)
	t.Run("rank with BM25", testRankWithBM25)
	t.Run("rank with vector search", testRankWithVector)
	t.Run("rank with hybrid search", testRankWithHybrid)
	t.Run("rank weight=0 has no effect", testWeightZero)
	t.Run("multiple conditions with weights", testMultipleConditions)
	t.Run("negative weight demotes", testNegativeWeightDemotes)
}

func setupSchema(t *testing.T) {
	t.Helper()
	helper.CreateClass(t, &models.Class{
		Class:      className,
		Vectorizer: "none",
		Properties: []*models.Property{
			{
				Name:            "name",
				DataType:        []string{"text"},
				Tokenization:    models.PropertyTokenizationWord,
				IndexFilterable: boolPtr(true),
				IndexSearchable: boolPtr(true),
			},
			{
				Name:            "price",
				DataType:        []string{"number"},
				IndexFilterable: boolPtr(true),
			},
			{
				Name:            "inStock",
				DataType:        []string{"boolean"},
				IndexFilterable: boolPtr(true),
			},
			{
				Name:            "category",
				DataType:        []string{"text"},
				Tokenization:    models.PropertyTokenizationField,
				IndexFilterable: boolPtr(true),
				IndexSearchable: boolPtr(true),
			},
			{
				Name:            "createdAt",
				DataType:        []string{"date"},
				IndexFilterable: boolPtr(true),
			},
		},
	})
}

func insertTestData(t *testing.T) {
	t.Helper()
	now := time.Now()
	objects := []*models.Object{
		{
			Class:  className,
			ID:     id1,
			Vector: []float32{1.0, 0.0, 0.0},
			Properties: map[string]interface{}{
				"name":      "Laptop Pro",
				"price":     float64(999),
				"inStock":   true,
				"category":  "electronics",
				"createdAt": now.Add(-1 * 24 * time.Hour).Format(time.RFC3339),
			},
		},
		{
			Class:  className,
			ID:     id2,
			Vector: []float32{0.9, 0.1, 0.0},
			Properties: map[string]interface{}{
				"name":      "Laptop Budget",
				"price":     float64(499),
				"inStock":   false,
				"category":  "electronics",
				"createdAt": now.Add(-30 * 24 * time.Hour).Format(time.RFC3339),
			},
		},
		{
			Class:  className,
			ID:     id3,
			Vector: []float32{0.0, 1.0, 0.0},
			Properties: map[string]interface{}{
				"name":      "Coffee Maker",
				"price":     float64(50),
				"inStock":   true,
				"category":  "kitchen",
				"createdAt": now.Add(-7 * 24 * time.Hour).Format(time.RFC3339),
			},
		},
		{
			Class:  className,
			ID:     id4,
			Vector: []float32{0.0, 0.0, 1.0},
			Properties: map[string]interface{}{
				"name":      "Running Shoes",
				"price":     float64(120),
				"inStock":   true,
				"category":  "sports",
				"createdAt": now.Add(-2 * 24 * time.Hour).Format(time.RFC3339),
			},
		},
		{
			Class:  className,
			ID:     id5,
			Vector: []float32{0.5, 0.5, 0.0},
			Properties: map[string]interface{}{
				"name":      "Tablet Standard",
				"price":     float64(350),
				"inStock":   false,
				"category":  "electronics",
				"createdAt": now.Add(-14 * 24 * time.Hour).Format(time.RFC3339),
			},
		},
	}

	helper.CreateObjectsBatch(t, objects)
}

func waitForIndexing(t *testing.T) {
	t.Helper()
	assert.EventuallyWithT(t, func(ct *assert.CollectT) {
		query := fmt.Sprintf(`{ Aggregate { %s { meta { count } } } }`, className)
		result := graphqlhelper.AssertGraphQL(t, helper.RootAuth, query)
		count := result.Get("Aggregate", className).AsSlice()
		require.NotEmpty(ct, count)
		meta := count[0].(map[string]interface{})["meta"].(map[string]interface{})
		assert.Equal(ct, float64(5), meta["count"])
	}, 10*time.Second, 500*time.Millisecond)
}

// testFilterOnlyRank tests rank with only filter conditions and no primary
// search signal. The rank score IS the ranking.
func testFilterOnlyRank(t *testing.T) {
	// Rank in-stock items higher — they should rank higher
	query := fmt.Sprintf(`{
		Get {
			%s(
				rank: {
					conditions: [{
						filter: {
							path: ["inStock"]
							operator: Equal
							valueBoolean: true
						}
					}]
					weight: 1.0
				}
			) {
				name
				inStock
				_additional { id }
			}
		}
	}`, className)

	result := graphqlhelper.AssertGraphQL(t, helper.RootAuth, query).
		Get("Get", className).
		AsSlice()

	require.Len(t, result, 5)

	// First 3 results should be in-stock items (score 1.0)
	for i := 0; i < 3; i++ {
		item := result[i].(map[string]interface{})
		assert.Equal(t, true, item["inStock"],
			"expected in-stock item at position %d, got %v", i, item["name"])
	}
	// Last 2 results should be out-of-stock items (score 0.0)
	for i := 3; i < 5; i++ {
		item := result[i].(map[string]interface{})
		assert.Equal(t, false, item["inStock"],
			"expected out-of-stock item at position %d, got %v", i, item["name"])
	}
}

// testDecayNumeric tests decay-based rank on a numeric property.
func testDecayNumeric(t *testing.T) {
	// Rank items with price near 100, using exponential decay with scale 200
	query := fmt.Sprintf(`{
		Get {
			%s(
				rank: {
					conditions: [{
						decay: {
							path: ["price"]
							origin: "100"
							scale: "200"
							curve: "exp"
						}
					}]
					weight: 1.0
				}
			) {
				name
				price
			}
		}
	}`, className)

	result := graphqlhelper.AssertGraphQL(t, helper.RootAuth, query).
		Get("Get", className).
		AsSlice()

	require.Len(t, result, 5)

	// Running Shoes (price=120, dist=20) should rank first — closest to origin 100
	first := result[0].(map[string]interface{})
	assert.Equal(t, "Running Shoes", first["name"])

	// Coffee Maker (price=50, dist=50) should rank second
	second := result[1].(map[string]interface{})
	assert.Equal(t, "Coffee Maker", second["name"])
}

// testRankWithBM25 tests rank fused with BM25 keyword search.
func testRankWithBM25(t *testing.T) {
	// BM25 search for "laptop" + rank in-stock items
	// Without rank: both laptops match equally well on "laptop"
	// With rank: in-stock laptop should rank higher
	query := fmt.Sprintf(`{
		Get {
			%s(
				bm25: { query: "laptop" }
				rank: {
					conditions: [{
						filter: {
							path: ["inStock"]
							operator: Equal
							valueBoolean: true
						}
					}]
					weight: 1.0
				}
			) {
				name
				inStock
			}
		}
	}`, className)

	result := graphqlhelper.AssertGraphQL(t, helper.RootAuth, query).
		Get("Get", className).
		AsSlice()

	// Should return laptop results, with in-stock ranked higher
	require.GreaterOrEqual(t, len(result), 2)

	// Laptop Pro (in stock) should rank higher than Laptop Budget (not in stock)
	first := result[0].(map[string]interface{})
	assert.Equal(t, "Laptop Pro", first["name"])
	assert.Equal(t, true, first["inStock"])
}

// testRankWithVector tests rank fused with vector search.
func testRankWithVector(t *testing.T) {
	// Vector search near electronics + rank in-stock
	query := fmt.Sprintf(`{
		Get {
			%s(
				nearVector: { vector: [1.0, 0.0, 0.0] }
				rank: {
					conditions: [{
						filter: {
							path: ["inStock"]
							operator: Equal
							valueBoolean: true
						}
					}]
					weight: 1.0
				}
			) {
				name
				inStock
			}
		}
	}`, className)

	result := graphqlhelper.AssertGraphQL(t, helper.RootAuth, query).
		Get("Get", className).
		AsSlice()

	require.NotEmpty(t, result)

	// First result should be Laptop Pro (closest vector AND in stock)
	first := result[0].(map[string]interface{})
	assert.Equal(t, "Laptop Pro", first["name"])
}

// testRankWithHybrid tests rank fused with hybrid (BM25 + vector) search.
func testRankWithHybrid(t *testing.T) {
	// Hybrid search for "laptop" + rank in-stock items
	// Both laptops match on keyword, but Laptop Pro is in stock
	query := fmt.Sprintf(`{
		Get {
			%s(
				hybrid: { query: "laptop", alpha: 0.5 }
				rank: {
					conditions: [{
						filter: {
							path: ["inStock"]
							operator: Equal
							valueBoolean: true
						}
					}]
					weight: 0.8
				}
			) {
				name
				inStock
			}
		}
	}`, className)

	result := graphqlhelper.AssertGraphQL(t, helper.RootAuth, query).
		Get("Get", className).
		AsSlice()

	require.GreaterOrEqual(t, len(result), 2)

	// Laptop Pro (in stock) should rank higher than Laptop Budget (not in stock)
	first := result[0].(map[string]interface{})
	assert.Equal(t, "Laptop Pro", first["name"])
	assert.Equal(t, true, first["inStock"])
}

// testWeightZero verifies that weight=0 means rank has no effect.
func testWeightZero(t *testing.T) {
	// Same query with and without rank at weight=0 should give same ordering
	baseQuery := fmt.Sprintf(`{
		Get {
			%s(
				nearVector: { vector: [1.0, 0.0, 0.0] }
			) {
				name
				_additional { id }
			}
		}
	}`, className)

	rankQuery := fmt.Sprintf(`{
		Get {
			%s(
				nearVector: { vector: [1.0, 0.0, 0.0] }
				rank: {
					conditions: [{
						filter: {
							path: ["inStock"]
							operator: Equal
							valueBoolean: false
						}
					}]
					weight: 0
				}
			) {
				name
				_additional { id }
			}
		}
	}`, className)

	baseResult := graphqlhelper.AssertGraphQL(t, helper.RootAuth, baseQuery).
		Get("Get", className).
		AsSlice()

	rankResult := graphqlhelper.AssertGraphQL(t, helper.RootAuth, rankQuery).
		Get("Get", className).
		AsSlice()

	require.NotEmpty(t, baseResult)
	require.NotEmpty(t, rankResult)

	// First result should be the same regardless of rank when weight=0
	baseFirst := baseResult[0].(map[string]interface{})["_additional"].(map[string]interface{})["id"]
	rankFirst := rankResult[0].(map[string]interface{})["_additional"].(map[string]interface{})["id"]
	assert.Equal(t, baseFirst, rankFirst, "weight=0 should not change ranking")
}

// testMultipleConditions tests rank with multiple weighted conditions.
func testMultipleConditions(t *testing.T) {
	// Rank: in-stock (weight 2) + price near 100 (weight 1)
	query := fmt.Sprintf(`{
		Get {
			%s(
				rank: {
					conditions: [
						{
							filter: {
								path: ["inStock"]
								operator: Equal
								valueBoolean: true
							}
							weight: 2.0
						},
						{
							decay: {
								path: ["price"]
								origin: "100"
								scale: "500"
								curve: "linear"
							}
							weight: 1.0
						}
					]
					weight: 1.0
				}
			) {
				name
				inStock
				price
			}
		}
	}`, className)

	result := graphqlhelper.AssertGraphQL(t, helper.RootAuth, query).
		Get("Get", className).
		AsSlice()

	require.Len(t, result, 5)

	// Running Shoes should rank first: in-stock (score=1, weight=2) + price near 100 (best decay, weight=1)
	first := result[0].(map[string]interface{})
	assert.Equal(t, "Running Shoes", first["name"])
}

// testNegativeWeightDemotes tests that negative per-condition weights demote matching results.
func testNegativeWeightDemotes(t *testing.T) {
	// Use negative weight to demote out-of-stock items
	query := fmt.Sprintf(`{
		Get {
			%s(
				rank: {
					conditions: [{
						filter: {
							path: ["inStock"]
							operator: Equal
							valueBoolean: false
						}
						weight: -1.0
					}]
					weight: 1.0
				}
			) {
				name
				inStock
				_additional { id }
			}
		}
	}`, className)

	result := graphqlhelper.AssertGraphQL(t, helper.RootAuth, query).
		Get("Get", className).
		AsSlice()

	require.Len(t, result, 5)

	// In-stock items should appear first (not demoted), out-of-stock last (demoted by negative weight)
	for i := 0; i < 3; i++ {
		item := result[i].(map[string]interface{})
		assert.Equal(t, true, item["inStock"],
			"expected in-stock item at position %d, got %v", i, item["name"])
	}
	for i := 3; i < 5; i++ {
		item := result[i].(map[string]interface{})
		assert.Equal(t, false, item["inStock"],
			"expected out-of-stock item at position %d, got %v", i, item["name"])
	}
}

func boolPtr(b bool) *bool {
	return &b
}
