model diffusion( x -> Omega, d -> NonNegative, y(x) ) { 
    dot(y) = d * div(grad(y, x), x) 
}

model logistic_growth(r -> NonNegative, k -> NonNegative, u(t) ) { 
    dot(u) = r * u * (1 - u / k)
}

model fishers(
    x -> [0 ... 1],
    t -> [0 ... 10],
    y(x, t)
) {
    use diffusion(x=x, d=1, y=y)
    use logistic_growth(r=1, k=1, u=y)
    y(x, 0) = 0
}
