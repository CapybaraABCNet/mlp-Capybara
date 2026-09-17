package main

import (
	"fmt"
	"math"
	"math/rand"
)

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidDeriv(x float64) float64 {
	return x * (1 - x)
}

func main() {
	w1 := rand.Float64()*2 - 1
	w2 := rand.Float64()*2 - 1
	w3 := rand.Float64()*2 - 1
	w4 := rand.Float64()*2 - 1

	w5 := rand.Float64()*2 - 1
	w6 := rand.Float64()*2 - 1

	b1 := rand.Float64()*2 - 1
	b2 := rand.Float64()*2 - 1
	b3 := rand.Float64()*2 - 1

	lr := 3.5

	inputs := [][]int{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := []int{0, 1, 1, 0}

	fmt.Println(w1, w2, w3, w4, w5, w6, b1, b2, b3)

	for epoch := 0; epoch < 10850; epoch++ {
		for i := range inputs {
			target := targets[i]

			x1 := inputs[i][0]
			x2 := inputs[i][1]

			sumH1 := float64(x1)*w1 + float64(x2)*w3 + b1
			sumH2 := float64(x1)*w2 + float64(x2)*w4 + b2

			h1 := sigmoid(sumH1)
			h2 := sigmoid(sumH2)

			sumY := h1*w5 + h2*w6 + b3

			y := sigmoid(sumY)

			errY := float64(target) - y
			deltaY := errY * sigmoidDeriv(y)

			errH1 := deltaY * w5
			errH2 := deltaY * w6

			deltaH1 := errH1 * sigmoidDeriv(h1)
			deltaH2 := errH2 * sigmoidDeriv(h2)

			w5 += lr * deltaY * float64(h1)
			w6 += lr * deltaY * float64(h2)
			b3 += lr * deltaY

			w1 += lr * deltaH1 * float64(x1)
			w2 += lr * deltaH2 * float64(x1)
			w3 += lr * deltaH1 * float64(x2)
			w4 += lr * deltaH2 * float64(x2)
			b1 += lr * deltaH1
			b2 += lr * deltaH2
		}
	}

	for j := range inputs {
		x1 := inputs[j][0]
		x2 := inputs[j][1]

		sumH1 := float64(x1)*w1 + float64(x2)*w3 + b1
		sumH2 := float64(x1)*w2 + float64(x2)*w4 + b2

		h1 := sigmoid(sumH1)
		h2 := sigmoid(sumH2)

		sumY := h1*w5 + h2*w6 + b3

		y := sigmoid(sumY)
		fmt.Printf("%d XOR %d = %.4f (ожидалось %d)\n", x1, x2, y, targets[j])
	}
}
