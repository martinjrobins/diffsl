model logistic_growth(r -> NonNegative, k -> NonNegative, y(t), z(t)) { 
    dot(y) = r * y * (1 - y / k)
    y(0) = 1.0
    z = 2 * y
}